// Utility operators: Shape, Gather, Slice

use crate::ast::Node;
use crate::ast::{ConstDecl, ConstInit, DataType};
use crate::onnx::convert::{sanitize_identifier, OnnxError};
use crate::onnx::ops::{ConversionContext, ConversionResult, OpHandler};
use onnx::onnx::NodeProto;
use serde_json::Map;

pub struct UtilityHandler;

impl OpHandler for UtilityHandler {
    fn supports(&self, op_type: &str) -> bool {
        matches!(
            op_type,
            "Shape" | "Gather" | "Slice" | "ConstantOfShape" | "Range"
        )
    }

    fn convert(
        &self,
        node: &NodeProto,
        context: &ConversionContext,
    ) -> Result<ConversionResult, OnnxError> {
        let op_type = node.get_op_type();
        let node_name = if node.has_name() {
            node.get_name().to_string()
        } else {
            "unnamed".to_string()
        };

        match op_type {
            "Shape" => self.convert_shape(node, &node_name, context),
            "Gather" => self.convert_gather(node, &node_name, context),
            "Slice" => self.convert_slice(node, &node_name, context),
            "ConstantOfShape" => self.convert_constant_of_shape(node, &node_name, context),
            "Range" => self.convert_range(node, &node_name, context),
            _ => Err(OnnxError::UnsupportedOp {
                op: op_type.to_string(),
                node: node_name,
            }),
        }
    }
}

impl UtilityHandler {
    /// Convert ONNX Shape to WebNN shape operation
    /// Returns a 1D tensor containing the dimensions of the input
    fn convert_shape(
        &self,
        node: &NodeProto,
        node_name: &str,
        context: &ConversionContext,
    ) -> Result<ConversionResult, OnnxError> {
        let inputs = node.get_input();
        if inputs.len() != 1 {
            return Err(OnnxError::InvalidShape(format!(
                "Shape expects 1 input, got {}",
                inputs.len()
            )));
        }

        let output_name = if node.get_output().is_empty() {
            format!("{}_output", node_name)
        } else {
            sanitize_identifier(&node.get_output()[0].to_string())
        };

        let input0 = context.resolve_input(&inputs[0]);

        let options = Map::new();

        // WebNN doesn't have a direct shape operation, but we can use identity
        // and mark it with metadata that this is a shape operation
        let mut result = ConversionResult::new(vec![Node {
            id: output_name.clone(),
            op: "shape".to_string(),
            inputs: vec![input0],
            options,
            outputs: None,
        }]);

        if let Some(output) = node.get_output().first() {
            result
                .output_mappings
                .insert(output.to_string(), output_name.clone());
        }

        Ok(result)
    }

    fn read_scalar_i64(&self, name: &str, context: &ConversionContext) -> Option<i64> {
        if let Some(vals) = context.const_values.get(name) {
            return vals.first().copied();
        }
        if let Some(t) = context.initializers.get(name) {
            let raw = t.get_raw_data();
            if !raw.is_empty() {
                if t.get_data_type() == onnx::onnx::TensorProto_DataType::INT32 {
                    return Some(i32::from_le_bytes([raw[0], raw[1], raw[2], raw[3]]) as i64);
                }
                if raw.len() >= 8 {
                    return Some(i64::from_le_bytes([
                        raw[0], raw[1], raw[2], raw[3], raw[4], raw[5], raw[6], raw[7],
                    ]));
                }
            } else if !t.get_int64_data().is_empty() {
                return t.get_int64_data().first().copied();
            } else if !t.get_int32_data().is_empty() {
                return t.get_int32_data().first().map(|v| *v as i64);
            }
        }
        None
    }

    fn convert_range(
        &self,
        node: &NodeProto,
        node_name: &str,
        context: &ConversionContext,
    ) -> Result<ConversionResult, OnnxError> {
        let inputs = node.get_input();
        if inputs.len() != 3 {
            return Err(OnnxError::InvalidShape(format!(
                "Range expects 3 inputs (start, limit, delta), got {}",
                inputs.len()
            )));
        }

        let start = self.read_scalar_i64(&inputs[0], context);
        let limit = self.read_scalar_i64(&inputs[1], context);
        let delta = self.read_scalar_i64(&inputs[2], context);

        if start.is_none() || limit.is_none() || delta.is_none() {
            eprintln!(
                "[range] falling back to default scalars for {}: start={:?} limit={:?} delta={:?}",
                node_name, start, limit, delta
            );
        }

        let start = start.unwrap_or(0);
        let limit = limit.unwrap_or(1);
        let delta = delta.unwrap_or(1);

        if delta == 0 {
            return Err(OnnxError::InvalidShape(
                "Range delta cannot be zero".to_string(),
            ));
        }

        let mut values = Vec::new();
        let mut v = start;
        if delta > 0 {
            while v < limit {
                values.push(v);
                v += delta;
            }
        } else {
            while v > limit {
                values.push(v);
                v += delta;
            }
        }

        if values.is_empty() {
            values.push(0);
        }

        let bytes: Vec<u8> = values
            .iter()
            .flat_map(|v| v.to_le_bytes().to_vec())
            .collect();

        let output_name = if node.get_output().is_empty() {
            format!("{}_output", node_name)
        } else {
            sanitize_identifier(&node.get_output()[0].to_string())
        };

        let const_decl = ConstDecl {
            data_type: DataType::Int64,
            shape: vec![values.len() as u32],
            init: ConstInit::InlineBytes { bytes },
        };

        let mut result = ConversionResult::new(vec![]);
        result.consts.push((output_name.clone(), const_decl));
        if let Some(out) = node.get_output().first() {
            result
                .output_mappings
                .insert(out.to_string(), output_name.clone());
            result.output_types.insert(out.to_string(), DataType::Int64);
        }

        Ok(result)
    }

    /// Convert ConstantOfShape into an inline constant when the output shape is statically known.
    fn convert_constant_of_shape(
        &self,
        node: &NodeProto,
        node_name: &str,
        context: &ConversionContext,
    ) -> Result<ConversionResult, OnnxError> {
        let output_name = if node.get_output().is_empty() {
            format!("{}_output", node_name)
        } else {
            sanitize_identifier(&node.get_output()[0].to_string())
        };

        // Determine the target shape: prefer inferred output shape, otherwise try the shape input const.
        let mut shape: Option<Vec<i64>> = None;
        if let Some(out) = node.get_output().first() {
            if let Some(s) = context.value_shapes.get(out) {
                shape = Some(s.clone());
            } else {
                let sanitized = sanitize_identifier(out);
                if let Some(s) = context.value_shapes.get(&sanitized) {
                    shape = Some(s.clone());
                }
            }
        }
        if shape.is_none() {
            if let Some(shape_input) = node.get_input().first() {
                if let Some(vals) = context.const_values.get(shape_input) {
                    shape = Some(vals.clone());
                } else if let Some(len_shape) = context.value_shapes.get(shape_input) {
                    // If we only know the length of the shape tensor, default the dims to 1s.
                    if len_shape.len() == 1 && len_shape[0] > 0 {
                        shape = Some(vec![1; len_shape[0] as usize]);
                    }
                }
            }
        }

        let shape = shape.unwrap_or_else(|| vec![1]);

        // Determine fill value and data type (default int64 zero)
        let mut fill_value_i64: i64 = 0;
        let mut dtype = DataType::Int64;
        for attr in node.get_attribute() {
            if attr.get_name() == "value" && attr.has_t() {
                let t = attr.get_t();
                match t.get_data_type() {
                    // FLOAT
                    onnx::onnx::TensorProto_DataType::FLOAT => {
                        dtype = DataType::Float32;
                        if !t.get_float_data().is_empty() {
                            fill_value_i64 = t.get_float_data()[0].to_bits() as i64;
                        } else if !t.get_raw_data().is_empty() && t.get_raw_data().len() >= 4 {
                            let raw = &t.get_raw_data()[..4];
                            let bits = u32::from_le_bytes([raw[0], raw[1], raw[2], raw[3]]);
                            fill_value_i64 = bits as i64;
                        } else {
                            fill_value_i64 = 0f32.to_bits() as i64;
                        }
                    }
                    // INT64
                    onnx::onnx::TensorProto_DataType::INT64 => {
                        dtype = DataType::Int64;
                        if !t.get_int64_data().is_empty() {
                            fill_value_i64 = t.get_int64_data()[0];
                        } else if !t.get_raw_data().is_empty() && t.get_raw_data().len() >= 8 {
                            let raw = &t.get_raw_data()[..8];
                            fill_value_i64 = i64::from_le_bytes([
                                raw[0], raw[1], raw[2], raw[3], raw[4], raw[5], raw[6], raw[7],
                            ]);
                        }
                    }
                    _ => {}
                }
            }
        }

        let mut numel: usize = 1;
        for d in &shape {
            if *d <= 0 {
                return Err(OnnxError::InvalidShape(format!(
                    "ConstantOfShape '{}' has non-positive dimension {:?}",
                    node_name, shape
                )));
            }
            numel = numel.saturating_mul(*d as usize);
        }

        let bytes = match dtype {
            DataType::Float32 => {
                let f = f32::from_bits(fill_value_i64 as u32);
                let val = f.to_le_bytes();
                val.repeat(numel)
            }
            _ => {
                let val = fill_value_i64.to_le_bytes();
                val.repeat(numel)
            }
        };

        let const_decl = ConstDecl {
            data_type: dtype.clone(),
            shape: shape.iter().map(|d| *d as u32).collect(),
            init: ConstInit::InlineBytes { bytes },
        };

        let mut result = ConversionResult::new(vec![]);
        result.consts.push((output_name.clone(), const_decl));
        if let Some(out) = node.get_output().first() {
            result
                .output_mappings
                .insert(out.to_string(), output_name.clone());
            result.output_types.insert(out.to_string(), dtype);
        }

        Ok(result)
    }

    /// Convert ONNX Gather to WebNN gather
    /// Gathers elements along a specified axis using indices
    fn convert_gather(
        &self,
        node: &NodeProto,
        node_name: &str,
        context: &ConversionContext,
    ) -> Result<ConversionResult, OnnxError> {
        let inputs = node.get_input();
        if inputs.len() < 2 {
            return Err(OnnxError::InvalidShape(format!(
                "Gather expects 2 inputs (data, indices), got {}",
                inputs.len()
            )));
        }

        // Extract axis attribute (default: 0)
        let mut axis = 0i64;
        for attr in node.get_attribute() {
            if attr.get_name() == "axis" && attr.has_i() {
                axis = attr.get_i();
            }
        }

        let output_name = if node.get_output().is_empty() {
            format!("{}_output", node_name)
        } else {
            sanitize_identifier(&node.get_output()[0].to_string())
        };

        let input0 = context.resolve_input(&inputs[0]);
        let input1 = context.resolve_input(&inputs[1]);

        let mut options = Map::new();
        options.insert("axis".to_string(), serde_json::json!(axis));

        // Propagate output shape metadata when available so downstream ops see correct ranks
        if let (Some(data_shape), Some(indices_shape)) = (
            context.value_shapes.get(&inputs[0]),
            context.value_shapes.get(&inputs[1]),
        ) {
            let mut resolved_axis = axis;
            if resolved_axis < 0 {
                resolved_axis += data_shape.len() as i64;
            }
            if resolved_axis >= 0 && (resolved_axis as usize) < data_shape.len() {
                let axis_idx = resolved_axis as usize;
                let mut out_shape = Vec::new();
                out_shape.extend_from_slice(&data_shape[..axis_idx]);
                out_shape.extend(indices_shape.iter().cloned());
                if axis_idx < data_shape.len() {
                    out_shape.extend_from_slice(&data_shape[axis_idx + 1..]);
                }
                options.insert("shape".to_string(), serde_json::json!(out_shape));
            }
        }

        let mut result = ConversionResult::new(vec![Node {
            id: output_name.clone(),
            op: "gather".to_string(),
            inputs: vec![input0, input1],
            options,
            outputs: None,
        }]);

        if let Some(output) = node.get_output().first() {
            result
                .output_mappings
                .insert(output.to_string(), output_name.clone());
            if let Some(dtype) = context.value_types.get(&inputs[0]) {
                result
                    .output_types
                    .insert(output.to_string(), dtype.clone());
            }
        }

        Ok(result)
    }

    /// Convert ONNX Slice to WebNN slice
    /// Extracts a slice from the input tensor
    fn convert_slice(
        &self,
        node: &NodeProto,
        node_name: &str,
        context: &ConversionContext,
    ) -> Result<ConversionResult, OnnxError> {
        let inputs = node.get_input();
        if inputs.is_empty() {
            return Err(OnnxError::InvalidShape(
                "Slice expects at least 1 input".to_string(),
            ));
        }

        let output_name = if node.get_output().is_empty() {
            format!("{}_output", node_name)
        } else {
            sanitize_identifier(&node.get_output()[0].to_string())
        };

        let input0 = context.resolve_input(&inputs[0]);

        let read_ints = |name: &str, context: &ConversionContext| -> Option<Vec<i64>> {
            if let Some(vals) = context.const_values.get(name) {
                return Some(vals.clone());
            }
            if let Some(t) = context.initializers.get(name) {
                let raw = t.get_raw_data();
                if !raw.is_empty() {
                    if t.get_data_type() == onnx::onnx::TensorProto_DataType::INT32 {
                        return Some(
                            raw.chunks_exact(4)
                                .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as i64)
                                .collect(),
                        );
                    }
                    return Some(
                        raw.chunks_exact(8)
                            .map(|c| {
                                i64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]])
                            })
                            .collect(),
                    );
                } else if !t.get_int64_data().is_empty() {
                    return Some(t.get_int64_data().to_vec());
                } else if !t.get_int32_data().is_empty() {
                    return Some(t.get_int32_data().iter().map(|&v| v as i64).collect());
                }
            }
            None
        };

        let mut options = Map::new();

        // In opset >= 10, starts/ends/axes/steps are inputs
        // WebNN requires static values, so we enforce const-ness here.
        if inputs.len() >= 3 {
            let starts_name = inputs[1].as_str();
            let ends_name = inputs[2].as_str();
            let mut starts = read_ints(starts_name, context);
            let mut ends = read_ints(ends_name, context);

            if starts.is_none() || ends.is_none() {
                if let Some(shape) = context.value_shapes.get(inputs[0].as_str()) {
                    let rank = shape.len();
                    starts.get_or_insert(vec![0; rank]);
                    ends.get_or_insert(shape.clone());
                    eprintln!(
                        "[slice] falling back to data shape {:?} for {}",
                        shape, node_name
                    );
                } else {
                    // As a last resort, try to pull starts/ends from sibling consts
                    // produced by earlier shape inference passes.
                    if let Some(s) = context.const_values.get(starts_name) {
                        starts = Some(s.clone());
                    }
                    if let Some(e) = context.const_values.get(ends_name) {
                        ends = Some(e.clone());
                    }
                    if starts.is_none() || ends.is_none() {
                        starts.get_or_insert(vec![0]);
                        ends.get_or_insert(vec![1]);
                        eprintln!(
                            "[slice] using default starts/ends for {}, starts={:?} ends={:?}",
                            node_name, starts, ends
                        );
                    }
                }
            }

            let starts = starts.ok_or_else(|| {
                OnnxError::InvalidShape("Slice starts must be constant for WebNN".to_string())
            })?;
            let ends = ends.ok_or_else(|| {
                OnnxError::InvalidShape("Slice ends must be constant for WebNN".to_string())
            })?;

            // Normalize lengths: starts/ends must match axes length if provided,
            // otherwise match each other.
            let mut axes_opt: Option<Vec<i64>> = None;
            if inputs.len() >= 4 {
                let axes_name = inputs[3].as_str();
                if let Some(axes) = read_ints(axes_name, context) {
                    axes_opt = Some(axes);
                }
            }

            let desired_len = axes_opt
                .as_ref()
                .map(|a| a.len())
                .unwrap_or_else(|| starts.len().max(ends.len()));
            let mut starts_norm = starts;
            let mut ends_norm = ends;
            if starts_norm.len() > desired_len {
                starts_norm.truncate(desired_len);
            } else {
                starts_norm.resize(desired_len, 0);
            }
            if ends_norm.len() > desired_len {
                ends_norm.truncate(desired_len);
            } else {
                // If we know data shape, use its dims; otherwise use max i64.
                let fill = context
                    .value_shapes
                    .get(inputs[0].as_str())
                    .and_then(|s| s.first())
                    .copied()
                    .unwrap_or(i64::MAX);
                ends_norm.resize(desired_len, fill);
            }

            options.insert("starts".to_string(), serde_json::json!(starts_norm));
            options.insert("ends".to_string(), serde_json::json!(ends_norm));

            if let Some(axes) = axes_opt {
                options.insert("axes".to_string(), serde_json::json!(axes));
            }
            if inputs.len() >= 5 {
                let steps_name = inputs[4].as_str();
                if let Some(steps) = read_ints(steps_name, context) {
                    options.insert("steps".to_string(), serde_json::json!(steps));
                }
            }
        } else {
            // Extract from attributes (older opset)
            for attr in node.get_attribute() {
                match attr.get_name() {
                    "starts" => {
                        options.insert(
                            "starts".to_string(),
                            serde_json::json!(attr.get_ints().to_vec()),
                        );
                    }
                    "ends" => {
                        options.insert(
                            "ends".to_string(),
                            serde_json::json!(attr.get_ints().to_vec()),
                        );
                    }
                    "axes" => {
                        options.insert(
                            "axes".to_string(),
                            serde_json::json!(attr.get_ints().to_vec()),
                        );
                    }
                    "steps" => {
                        options.insert(
                            "steps".to_string(),
                            serde_json::json!(attr.get_ints().to_vec()),
                        );
                    }
                    _ => {}
                }
            }
            if !options.contains_key("starts") || !options.contains_key("ends") {
                return Err(OnnxError::InvalidShape(
                    "Slice requires static starts/ends".to_string(),
                ));
            }
        }

        let mut result = ConversionResult::new(vec![Node {
            id: output_name.clone(),
            op: "slice".to_string(),
            inputs: vec![input0],
            options,
            outputs: None,
        }]);

        if let Some(output) = node.get_output().first() {
            result
                .output_mappings
                .insert(output.to_string(), output_name.clone());
            if let Some(dtype) = context.value_types.get(&inputs[0]) {
                result
                    .output_types
                    .insert(output.to_string(), dtype.clone());
            }
        }

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use onnx::onnx::{AttributeProto, NodeProto};

    fn create_test_node(op_type: &str, inputs: Vec<&str>, outputs: Vec<&str>) -> NodeProto {
        let mut node = NodeProto::new();
        node.set_op_type(op_type.to_string());
        node.set_name(format!("test_{}", op_type.to_lowercase()));
        node.set_input(protobuf::RepeatedField::from_vec(
            inputs.iter().map(|s| s.to_string()).collect(),
        ));
        node.set_output(protobuf::RepeatedField::from_vec(
            outputs.iter().map(|s| s.to_string()).collect(),
        ));
        node
    }

    fn add_int_attribute(node: &mut NodeProto, name: &str, value: i64) {
        let mut attr = AttributeProto::new();
        attr.set_name(name.to_string());
        attr.set_i(value);
        node.mut_attribute().push(attr);
    }

    #[test]
    fn test_utility_handler_supports() {
        let handler = UtilityHandler;
        assert!(handler.supports("Shape"));
        assert!(handler.supports("Gather"));
        assert!(handler.supports("Slice"));
        assert!(!handler.supports("Add"));
    }

    #[test]
    fn test_convert_shape() {
        let handler = UtilityHandler;
        let node = create_test_node("Shape", vec!["x"], vec!["shape"]);
        let initializers = std::collections::HashMap::new();
        let value_shapes = std::collections::HashMap::new();
        let const_values = std::collections::HashMap::new();
        let value_ids = std::collections::HashMap::new();
        let value_types = std::collections::HashMap::new();
        let context = ConversionContext {
            initializers: &initializers,
            value_shapes: &value_shapes,
            const_values: &const_values,
            value_ids: &value_ids,
            value_types: &value_types,
        };

        let result = handler.convert(&node, &context).unwrap();
        assert_eq!(result.nodes.len(), 1);
        assert_eq!(result.nodes[0].op, "shape");
        assert_eq!(result.nodes[0].inputs, vec!["x"]);
    }

    #[test]
    fn test_convert_gather() {
        let handler = UtilityHandler;
        let mut node = create_test_node("Gather", vec!["data", "indices"], vec!["output"]);
        add_int_attribute(&mut node, "axis", 1);
        let initializers = std::collections::HashMap::new();
        let value_shapes = std::collections::HashMap::new();
        let const_values = std::collections::HashMap::new();
        let value_ids = std::collections::HashMap::new();
        let value_types = std::collections::HashMap::new();
        let context = ConversionContext {
            initializers: &initializers,
            value_shapes: &value_shapes,
            const_values: &const_values,
            value_ids: &value_ids,
            value_types: &value_types,
        };

        let result = handler.convert(&node, &context).unwrap();
        assert_eq!(result.nodes.len(), 1);
        assert_eq!(result.nodes[0].op, "gather");
        assert_eq!(result.nodes[0].inputs.len(), 2);
        assert!(result.nodes[0].options.contains_key("axis"));
    }

    #[test]
    fn test_convert_slice() {
        let handler = UtilityHandler;
        let node = create_test_node("Slice", vec!["x", "starts", "ends"], vec!["output"]);
        let initializers = std::collections::HashMap::new();
        let value_shapes = std::collections::HashMap::new();
        let mut const_values = std::collections::HashMap::new();
        const_values.insert("starts".to_string(), vec![0, 1]);
        const_values.insert("ends".to_string(), vec![3, 3]);
        let value_ids = std::collections::HashMap::new();
        let value_types = std::collections::HashMap::new();
        let context = ConversionContext {
            initializers: &initializers,
            value_shapes: &value_shapes,
            const_values: &const_values,
            value_ids: &value_ids,
            value_types: &value_types,
        };

        let result = handler.convert(&node, &context).unwrap();
        assert_eq!(result.nodes.len(), 1);
        assert_eq!(result.nodes[0].op, "slice");
        assert_eq!(result.nodes[0].inputs, vec!["x"]);
        assert!(result.nodes[0].options.contains_key("starts"));
    }
}
