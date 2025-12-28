// Static shape/type inference scaffold for ONNX graphs.
// Conservative: records only fully-static shapes and folds small integer constants
// to unblock reshape/axes/starts/ends calculations. Dynamic dims cause errors so
// callers can ask users to run onnx-simplifier or provide overrides.
use crate::ast::DataType;
use crate::onnx::ir::{Dim, OnnxIrGraph, TensorShape, TensorType};
use crate::onnx::types::map_onnx_data_type;
use onnx::onnx::{GraphProto, ModelProto, NodeProto, TensorProto, TensorProto_DataType};
use std::collections::{HashMap, HashSet};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ShapeInferenceError {
    #[error("input '{0}' is missing shape information")]
    MissingInputShape(String),
    #[error("input '{input}' has dynamic dimension '{dim}', please provide an override")]
    DynamicDim { input: String, dim: String },
    #[error("unsupported ONNX data type: {0}")]
    UnsupportedDataType(i32),
    #[error("could not infer shape for op '{op}'")]
    CannotInfer { op: String },
}

#[derive(Debug, Default)]
pub struct InferenceResult {
    pub value_shapes: HashMap<String, Vec<i64>>,
    pub value_types: HashMap<String, DataType>,
    pub const_values: HashMap<String, Vec<i64>>,
}

/// Run a lightweight static shape/type inference pass.
/// Returns only fully-known shapes; dynamic dimensions trigger an error.
pub fn infer_static_shapes(
    model: &ModelProto,
    overrides: &HashMap<String, u32>,
) -> Result<InferenceResult, ShapeInferenceError> {
    let mut result = InferenceResult::default();

    if !model.has_graph() {
        return Ok(result);
    }

    let graph = model.get_graph();
    let mut ir = OnnxIrGraph::default();
    let initializer_names: HashSet<String> = graph
        .get_initializer()
        .iter()
        .map(|i| i.get_name().to_string())
        .collect();

    seed_inputs(graph, overrides, &initializer_names, &mut ir, &mut result)?;
    seed_initializers(graph, &mut ir, &mut result)?;
    seed_constant_nodes(graph, &mut result, &mut ir)?;

    propagate_node_shapes(graph, &mut result)?;

    Ok(result)
}

fn seed_inputs(
    graph: &GraphProto,
    overrides: &HashMap<String, u32>,
    initializer_names: &HashSet<String>,
    ir: &mut OnnxIrGraph,
    result: &mut InferenceResult,
) -> Result<(), ShapeInferenceError> {
    for input in graph.get_input() {
        let name = input.get_name().to_string();
        let vi = ir.value_or_insert(&name);
        vi.producer = None;

        if initializer_names.contains(&name) {
            continue;
        }

        if !input.has_field_type() || !input.get_field_type().has_tensor_type() {
            return Err(ShapeInferenceError::MissingInputShape(name));
        }

        let tensor_type = input.get_field_type().get_tensor_type();
        let dtype = if tensor_type.has_elem_type() {
            map_onnx_data_type(tensor_type.get_elem_type() as i32).map_err(|_| {
                ShapeInferenceError::UnsupportedDataType(tensor_type.get_elem_type() as i32)
            })?
        } else {
            return Err(ShapeInferenceError::UnsupportedDataType(0));
        };

        if !tensor_type.has_shape() {
            return Err(ShapeInferenceError::MissingInputShape(name));
        }

        let mut dims = Vec::new();
        for dim in tensor_type.get_shape().get_dim() {
            if dim.has_dim_value() {
                dims.push(Dim::Known(dim.get_dim_value()));
            } else if dim.has_dim_param() {
                let key = dim.get_dim_param().to_string();
                if let Some(v) = overrides.get(&key) {
                    dims.push(Dim::Known(*v as i64));
                } else {
                    return Err(ShapeInferenceError::DynamicDim {
                        input: name.clone(),
                        dim: key,
                    });
                }
            } else {
                return Err(ShapeInferenceError::MissingInputShape(name));
            }
        }

        let ty = TensorType {
            data_type: dtype.clone(),
            shape: TensorShape { dims },
        };
        vi.ty = Some(ty.clone());
        result.value_types.insert(name.clone(), dtype);
        if let Some(shape) = ty.shape.to_i64() {
            result.value_shapes.insert(name, shape);
        }
    }
    Ok(())
}

fn seed_initializers(
    graph: &GraphProto,
    ir: &mut OnnxIrGraph,
    result: &mut InferenceResult,
) -> Result<(), ShapeInferenceError> {
    for init in graph.get_initializer() {
        let name = init.get_name().to_string();
        let vi = ir.value_or_insert(&name);
        vi.producer = None;

        let dtype = map_onnx_data_type(init.get_data_type() as i32)
            .map_err(|_| ShapeInferenceError::UnsupportedDataType(init.get_data_type() as i32))?;
        let shape: Vec<i64> = init.get_dims().to_vec();
        result.value_types.insert(name.clone(), dtype.clone());
        result.value_shapes.insert(name.clone(), shape);

        if matches!(
            dtype,
            DataType::Int32 | DataType::Int64 | DataType::Uint32 | DataType::Uint64
        ) {
            let values = read_int_tensor(init);
            if !values.is_empty() {
                result.const_values.insert(name, values);
            }
        }
    }
    Ok(())
}

fn seed_constant_nodes(
    graph: &GraphProto,
    result: &mut InferenceResult,
    ir: &mut OnnxIrGraph,
) -> Result<(), ShapeInferenceError> {
    for node in graph.get_node() {
        if node.get_op_type() != "Constant" {
            continue;
        }

        if let Some(out) = node.get_output().first() {
            let out_name = out.to_string();
            let vi = ir.value_or_insert(&out_name);
            vi.producer = Some(node.get_name().to_string());

            if let Some(attr) = node
                .get_attribute()
                .iter()
                .find(|a| a.get_name() == "value" && a.has_t())
            {
                let t = attr.get_t();
                let dtype = map_onnx_data_type(t.get_data_type() as i32).map_err(|_| {
                    ShapeInferenceError::UnsupportedDataType(t.get_data_type() as i32)
                })?;
                result.value_types.insert(out_name.clone(), dtype);

                let vals = read_int_tensor(t);
                if !vals.is_empty() {
                    result.const_values.insert(out_name.clone(), vals.clone());
                    let shape: Vec<i64> = if vals.len() == 1 {
                        Vec::new()
                    } else {
                        vec![vals.len() as i64]
                    };
                    result.value_shapes.insert(out_name.clone(), shape);
                    vi.ty = Some(TensorType {
                        data_type: result.value_types[&out_name].clone(),
                        shape: TensorShape::from_known(result.value_shapes[&out_name].clone()),
                    });
                }
            }
        }
    }
    Ok(())
}

fn propagate_node_shapes(
    graph: &GraphProto,
    result: &mut InferenceResult,
) -> Result<(), ShapeInferenceError> {
    let mut progress = true;
    let max_iters = 8;
    let mut iter = 0;

    while progress && iter < max_iters {
        progress = false;
        iter += 1;

        for node in graph.get_node() {
            let outputs = node.get_output();
            if outputs.is_empty() {
                continue;
            }
            if outputs
                .iter()
                .all(|o| result.value_shapes.contains_key(o.as_str()))
            {
                continue;
            }

            if let Some(shape) = infer_node_shape(node, result) {
                let out_name = outputs[0].to_string();
                result.value_shapes.entry(out_name.clone()).or_insert(shape);

                // Propagate dtype from first input if available.
                if let Some(first_in) = node.get_input().first() {
                    if let Some(dtype) = result.value_types.get(first_in).cloned() {
                        result.value_types.entry(out_name.clone()).or_insert(dtype);
                    }
                }

                progress = true;
            }
        }

        // Opportunistic const folding for integer tensors to unlock more shapes.
        progress |= fold_integer_constants(graph, result);
    }

    Ok(())
}

fn broadcast_shapes(a: &[i64], b: &[i64]) -> Option<Vec<i64>> {
    let mut result = Vec::new();
    let mut ai = a.iter().rev();
    let mut bi = b.iter().rev();

    loop {
        match (ai.next(), bi.next()) {
            (Some(&ad), Some(&bd)) => {
                if ad == bd {
                    result.push(ad);
                } else if ad == 1 {
                    result.push(bd);
                } else if bd == 1 {
                    result.push(ad);
                } else {
                    return None;
                }
            }
            (Some(&ad), None) => result.push(ad),
            (None, Some(&bd)) => result.push(bd),
            (None, None) => break,
        }
    }

    result.reverse();
    Some(result)
}

fn infer_node_shape(node: &NodeProto, ctx: &InferenceResult) -> Option<Vec<i64>> {
    let op = node.get_op_type();
    match op {
        "Relu" | "Tanh" | "Sigmoid" | "Erf" | "Softmax" | "Gelu" | "Exp" | "Log" | "Abs"
        | "Neg" | "Sqrt" | "LayerNormalization" => node
            .get_input()
            .first()
            .and_then(|i| ctx.value_shapes.get(i).cloned()),
        "Add" | "Sub" | "Mul" | "Div" | "Pow" => {
            if node.get_input().len() < 2 {
                return None;
            }
            let a = node.get_input()[0].as_str();
            let b = node.get_input()[1].as_str();
            match (ctx.value_shapes.get(a), ctx.value_shapes.get(b)) {
                (Some(sa), Some(sb)) => broadcast_shapes(sa, sb),
                _ => None,
            }
        }
        "MatMul" => {
            if node.get_input().len() < 2 {
                return None;
            }
            let a_shape = ctx.value_shapes.get(node.get_input()[0].as_str())?;
            let b_shape = ctx.value_shapes.get(node.get_input()[1].as_str())?;

            // Attention pattern: rank-4 [B,S,H,D] x [B,S,H,D] -> [B,S,H,H]
            if a_shape.len() == 4 && b_shape.len() == 4 {
                return Some(vec![a_shape[0], a_shape[1], a_shape[2], b_shape[3]]);
            }

            // Fallback generic matmul
            if a_shape.len() >= 2 && b_shape.len() >= 2 {
                let m = a_shape[a_shape.len() - 2];
                let n = b_shape[b_shape.len() - 1];
                let mut out = Vec::new();
                if a_shape.len() > 2 {
                    out.extend_from_slice(&a_shape[..a_shape.len() - 2]);
                }
                out.push(m);
                out.push(n);
                return Some(out);
            }
            None
        }
        "Transpose" => {
            let input = node.get_input().first()?;
            let shape = ctx.value_shapes.get(input)?;
            let perm: Vec<usize> = node
                .get_attribute()
                .iter()
                .find(|a| a.get_name() == "perm")
                .map(|a| a.get_ints().iter().map(|&i| i as usize).collect())
                .unwrap_or_else(|| (0..shape.len()).rev().collect());
            if perm.iter().any(|&i| i >= shape.len()) {
                return None;
            }
            Some(perm.iter().map(|&i| shape[i]).collect())
        }
        "Concat" => {
            let mut shapes = Vec::new();
            for inp in node.get_input() {
                if let Some(s) = ctx.value_shapes.get(inp.as_str()) {
                    shapes.push(s.clone());
                } else {
                    return None;
                }
            }
            if shapes.is_empty() {
                return None;
            }
            let mut axis = node
                .get_attribute()
                .iter()
                .find(|a| a.get_name() == "axis" && a.has_i())
                .map(|a| a.get_i())
                .unwrap_or(0);
            if axis < 0 {
                axis += shapes[0].len() as i64;
            }
            let axis = axis as usize;
            let mut out = shapes[0].clone();
            for s in shapes.iter().skip(1) {
                if s.len() != out.len() || axis >= s.len() {
                    return None;
                }
                out[axis] += s[axis];
            }
            Some(out)
        }
        "Unsqueeze" => {
            if node.get_input().is_empty() {
                return None;
            }
            let input_shape = ctx.value_shapes.get(node.get_input()[0].as_str())?;
            let axes = node
                .get_attribute()
                .iter()
                .find(|a| a.get_name() == "axes")
                .map(|a| a.get_ints().to_vec())
                .unwrap_or_default();
            if axes.is_empty() {
                return None;
            }
            let mut output_shape = input_shape.clone();
            let mut sorted_axes = axes.clone();
            sorted_axes.sort();
            for axis in sorted_axes {
                let idx = if axis < 0 {
                    (output_shape.len() as i64 + axis + 1) as usize
                } else {
                    axis as usize
                };
                if idx > output_shape.len() {
                    return None;
                }
                output_shape.insert(idx, 1);
            }
            Some(output_shape)
        }
        "Squeeze" => {
            if node.get_input().is_empty() {
                return None;
            }
            let input_shape = ctx.value_shapes.get(node.get_input()[0].as_str())?;
            let axes = node
                .get_attribute()
                .iter()
                .find(|a| a.get_name() == "axes")
                .map(|a| a.get_ints().to_vec())
                .unwrap_or_default();
            let mut output_shape = input_shape.clone();
            if axes.is_empty() {
                output_shape.retain(|&d| d != 1);
                return Some(output_shape);
            }
            let mut axes_norm: Vec<usize> = axes
                .iter()
                .map(|&a| {
                    if a < 0 {
                        (input_shape.len() as i64 + a) as usize
                    } else {
                        a as usize
                    }
                })
                .collect();
            axes_norm.sort();
            axes_norm.dedup();
            let mut keep = Vec::new();
            for (idx, dim) in input_shape.iter().enumerate() {
                if axes_norm.contains(&idx) {
                    continue;
                }
                keep.push(*dim);
            }
            Some(keep)
        }
        "Reshape" => {
            if node.get_input().len() < 2 {
                return None;
            }
            let data_shape = ctx.value_shapes.get(node.get_input()[0].as_str())?;
            let shape_input = node.get_input()[1].as_str();
            let mut target: Vec<i64> = ctx.const_values.get(shape_input)?.clone();

            if target.contains(&-1) {
                let total_input: i64 = data_shape.iter().product();
                let known: i64 = target.iter().filter(|&&d| d != -1).product();
                if known == 0 || total_input % known != 0 {
                    return None;
                }
                if let Some(idx) = target.iter().position(|&d| d == -1) {
                    target[idx] = total_input / known;
                }
            }
            Some(target)
        }
        "Slice" => {
            if node.get_input().is_empty() {
                return None;
            }
            let data_shape = ctx.value_shapes.get(node.get_input()[0].as_str())?;
            let starts = node
                .get_input()
                .get(1)
                .and_then(|n| ctx.const_values.get(n))
                .cloned()?;
            let ends = node
                .get_input()
                .get(2)
                .and_then(|n| ctx.const_values.get(n))
                .cloned()?;
            let axes = node
                .get_input()
                .get(3)
                .and_then(|n| ctx.const_values.get(n))
                .cloned()
                .unwrap_or_else(|| (0..data_shape.len() as i64).collect());
            let steps = node
                .get_input()
                .get(4)
                .and_then(|n| ctx.const_values.get(n))
                .cloned()
                .unwrap_or_else(|| vec![1; axes.len()]);

            if axes.len() != starts.len() || axes.len() != ends.len() || axes.len() != steps.len() {
                return None;
            }

            let mut out = data_shape.clone();
            for i in 0..axes.len() {
                let mut axis = axes[i];
                if axis < 0 {
                    axis += data_shape.len() as i64;
                }
                let axis = axis as usize;
                if axis >= out.len() {
                    return None;
                }
                if steps[i] != 1 {
                    return None;
                }
                let dim = data_shape[axis];
                let mut start = starts[i];
                let mut end = ends[i];
                if start < 0 {
                    start += dim;
                }
                if end < 0 {
                    end += dim;
                }
                start = start.max(0);
                end = end.min(dim);
                out[axis] = if end < start { 0 } else { end - start };
            }
            Some(out)
        }
        "Gather" => {
            if node.get_input().len() < 2 {
                return None;
            }
            let data_shape = ctx.value_shapes.get(node.get_input()[0].as_str())?;
            let indices_shape = ctx.value_shapes.get(node.get_input()[1].as_str())?;
            let mut axis = node
                .get_attribute()
                .iter()
                .find(|a| a.get_name() == "axis" && a.has_i())
                .map(|a| a.get_i())
                .unwrap_or(0);
            if axis < 0 {
                axis += data_shape.len() as i64;
            }
            let axis = axis as usize;
            if axis > data_shape.len() {
                return None;
            }
            let mut out = Vec::new();
            out.extend_from_slice(&data_shape[..axis]);
            out.extend(indices_shape.iter().cloned());
            if axis < data_shape.len() {
                out.extend_from_slice(&data_shape[axis + 1..]);
            }
            Some(out)
        }
        "Split" => {
            let input_shape = node
                .get_input()
                .first()
                .and_then(|i| ctx.value_shapes.get(i))
                .cloned()?;
            let mut axis = node
                .get_attribute()
                .iter()
                .find(|a| a.get_name() == "axis" && a.has_i())
                .map(|a| a.get_i())
                .unwrap_or(0);
            if axis < 0 {
                axis += input_shape.len() as i64;
            }
            let axis = axis as usize;
            if axis >= input_shape.len() {
                return None;
            }
            let splits = node
                .get_attribute()
                .iter()
                .find(|a| a.get_name() == "split")
                .map(|a| a.get_ints().to_vec());
            if let Some(s) = splits {
                if s.iter().any(|&v| v <= 0) {
                    return None;
                }
                let sum: i64 = s.iter().sum();
                if sum != input_shape[axis] {
                    return None;
                }
                let mut out = input_shape.clone();
                out[axis] = s[0];
                Some(out)
            } else {
                let outputs = node.get_output().len() as i64;
                if outputs == 0 || input_shape[axis] % outputs != 0 {
                    return None;
                }
                let chunk = input_shape[axis] / outputs;
                let mut out = input_shape.clone();
                out[axis] = chunk;
                Some(out)
            }
        }
        "ReduceMean" | "ReduceSum" | "ReduceMax" | "ReduceMin" => {
            let input = node.get_input().first()?;
            let input_shape = ctx.value_shapes.get(input)?;
            let axes: Vec<i64> = node
                .get_attribute()
                .iter()
                .find(|a| a.get_name() == "axes")
                .map(|a| a.get_ints().to_vec())
                .unwrap_or_default();
            let keepdims = node
                .get_attribute()
                .iter()
                .find(|a| a.get_name() == "keepdims" && a.has_i())
                .map(|a| a.get_i() != 0)
                .unwrap_or(true);
            if axes.is_empty() {
                if keepdims {
                    Some(vec![1; input_shape.len()])
                } else {
                    Some(vec![])
                }
            } else {
                let mut out = input_shape.clone();
                for axis in axes {
                    let mut a = axis;
                    if a < 0 {
                        a += input_shape.len() as i64;
                    }
                    let idx = a as usize;
                    if idx >= out.len() {
                        return None;
                    }
                    if keepdims {
                        out[idx] = 1;
                    } else {
                        out[idx] = -1;
                    }
                }
                if !keepdims {
                    out.retain(|&d| d != -1);
                }
                Some(out)
            }
        }
        _ => None,
    }
}

fn fold_integer_constants(graph: &GraphProto, ctx: &mut InferenceResult) -> bool {
    let mut changed = false;
    for node in graph.get_node() {
        let outputs = node.get_output();
        if outputs.is_empty() {
            continue;
        }
        if ctx.const_values.contains_key(outputs[0].as_str()) {
            continue;
        }

        let op = node.get_op_type();
        let inputs = node.get_input();
        let all_const = inputs
            .iter()
            .all(|i| ctx.const_values.contains_key(i.as_str()));
        if !all_const {
            continue;
        }

        match op {
            "Shape" => {
                if let Some(inp) = inputs.first() {
                    if let Some(shape) = ctx.value_shapes.get(inp.as_str()) {
                        let out_name = outputs[0].to_string();
                        ctx.const_values.insert(out_name.clone(), shape.clone());
                        ctx.value_shapes.insert(out_name, vec![shape.len() as i64]);
                        changed = true;
                    }
                }
            }
            "Concat" => {
                let mut axis = 0i64;
                for attr in node.get_attribute() {
                    if attr.get_name() == "axis" && attr.has_i() {
                        axis = attr.get_i();
                    }
                }
                if axis == 0 {
                    let mut combined = Vec::new();
                    for inp in inputs {
                        if let Some(vals) = ctx.const_values.get(inp.as_str()) {
                            combined.extend_from_slice(vals);
                        }
                    }
                    if !combined.is_empty() {
                        let out_name = outputs[0].to_string();
                        ctx.const_values.insert(out_name.clone(), combined.clone());
                        ctx.value_shapes
                            .insert(out_name, vec![combined.len() as i64]);
                        changed = true;
                    }
                }
            }
            "Gather" => {
                let mut axis = 0i64;
                for attr in node.get_attribute() {
                    if attr.get_name() == "axis" && attr.has_i() {
                        axis = attr.get_i();
                    }
                }
                if axis == 0 && inputs.len() >= 2 {
                    let data = ctx.const_values.get(inputs[0].as_str());
                    let indices = ctx.const_values.get(inputs[1].as_str());
                    if let (Some(data), Some(indices)) = (data, indices) {
                        let mut gathered = Vec::new();
                        for &idx in indices {
                            let i = if idx < 0 {
                                (data.len() as i64 + idx) as usize
                            } else {
                                idx as usize
                            };
                            if let Some(v) = data.get(i) {
                                gathered.push(*v);
                            }
                        }
                        if !gathered.is_empty() {
                            let out_name = outputs[0].to_string();
                            ctx.const_values.insert(out_name.clone(), gathered.clone());
                            let shape = if gathered.len() == 1 {
                                Vec::new()
                            } else {
                                vec![gathered.len() as i64]
                            };
                            ctx.value_shapes.insert(out_name, shape);
                            changed = true;
                        }
                    }
                }
            }
            _ => {}
        }
    }
    changed
}

fn read_int_tensor(tensor: &TensorProto) -> Vec<i64> {
    let raw = tensor.get_raw_data();
    if !raw.is_empty() {
        match tensor.get_data_type() {
            TensorProto_DataType::INT32 => raw
                .chunks_exact(4)
                .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as i64)
                .collect(),
            _ => raw
                .chunks_exact(8)
                .map(|c| i64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]))
                .collect(),
        }
    } else if !tensor.get_int64_data().is_empty() {
        tensor.get_int64_data().to_vec()
    } else if !tensor.get_int32_data().is_empty() {
        tensor.get_int32_data().iter().map(|&v| v as i64).collect()
    } else {
        Vec::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dynamic_dim_requires_override() {
        let mut dim = onnx::onnx::TensorShapeProto_Dimension::new();
        dim.set_dim_param("batch".to_string());
        let mut shape = onnx::onnx::TensorShapeProto::new();
        shape.mut_dim().push(dim);

        let mut tensor_type = onnx::onnx::TypeProto_Tensor::new();
        tensor_type.set_elem_type(onnx::onnx::TensorProto_DataType::FLOAT);
        tensor_type.set_shape(shape);

        let mut type_proto = onnx::onnx::TypeProto::new();
        type_proto.set_tensor_type(tensor_type);

        let mut vi = onnx::onnx::ValueInfoProto::new();
        vi.set_name("input".to_string());
        vi.set_field_type(type_proto);

        let mut graph = onnx::onnx::GraphProto::new();
        graph.mut_input().push(vi);

        let mut model = onnx::onnx::ModelProto::new();
        model.set_graph(graph);

        let res = infer_static_shapes(&model, &HashMap::new());
        assert!(matches!(
            res,
            Err(ShapeInferenceError::DynamicDim { dim, .. }) if dim == "batch"
        ));
    }

    #[test]
    fn override_allows_static_shape() {
        let mut dim = onnx::onnx::TensorShapeProto_Dimension::new();
        dim.set_dim_param("batch".to_string());
        let mut shape = onnx::onnx::TensorShapeProto::new();
        shape.mut_dim().push(dim);

        let mut tensor_type = onnx::onnx::TypeProto_Tensor::new();
        tensor_type.set_elem_type(onnx::onnx::TensorProto_DataType::FLOAT);
        tensor_type.set_shape(shape);

        let mut type_proto = onnx::onnx::TypeProto::new();
        type_proto.set_tensor_type(tensor_type);

        let mut vi = onnx::onnx::ValueInfoProto::new();
        vi.set_name("input".to_string());
        vi.set_field_type(type_proto);

        let mut graph = onnx::onnx::GraphProto::new();
        graph.mut_input().push(vi);

        let mut model = onnx::onnx::ModelProto::new();
        model.set_graph(graph);

        let mut overrides = HashMap::new();
        overrides.insert("batch".to_string(), 1);
        let res = infer_static_shapes(&model, &overrides).unwrap();
        assert_eq!(res.value_shapes.get("input"), Some(&vec![1]));
    }
}
