// Range operation evaluator
// Generates a sequence of numbers from start to end with given delta

use crate::onnx::constant_folding::{
    ConstantEvaluator as EvaluatorTrait, ConstantFoldingContext, ConstantTensor, TensorData,
};
use crate::onnx::convert::OnnxError;
use onnx::onnx::{NodeProto, TensorProto_DataType};

pub struct RangeEvaluator;

impl EvaluatorTrait for RangeEvaluator {
    fn op_type(&self) -> &str {
        "Range"
    }

    fn can_evaluate(&self, node: &NodeProto, ctx: &ConstantFoldingContext) -> bool {
        if node.get_op_type() != "Range" {
            return false;
        }

        // Range requires 3 constant inputs: start, limit (end), delta (step)
        if node.get_input().len() != 3 {
            return false;
        }

        // All inputs must be constants
        node.get_input()
            .iter()
            .all(|inp| ctx.is_constant(inp.as_str()))
    }

    fn evaluate(
        &self,
        node: &NodeProto,
        ctx: &ConstantFoldingContext,
    ) -> Result<Vec<ConstantTensor>, OnnxError> {
        if node.get_input().len() != 3 {
            return Err(OnnxError::MissingAttribute {
                attr: "inputs (need 3: start, limit, delta)".to_string(),
                op: "Range".to_string(),
            });
        }

        let start_name = &node.get_input()[0];
        let limit_name = &node.get_input()[1];
        let delta_name = &node.get_input()[2];

        let start_tensor = ctx.get_constant(start_name.as_str()).ok_or_else(|| {
            OnnxError::ShapeInference(format!("Start tensor '{}' not found", start_name))
        })?;

        let limit_tensor = ctx.get_constant(limit_name.as_str()).ok_or_else(|| {
            OnnxError::ShapeInference(format!("Limit tensor '{}' not found", limit_name))
        })?;

        let delta_tensor = ctx.get_constant(delta_name.as_str()).ok_or_else(|| {
            OnnxError::ShapeInference(format!("Delta tensor '{}' not found", delta_name))
        })?;

        // Extract scalar values - Range inputs should be scalar tensors
        let start = extract_scalar_i64(&start_tensor.data)?;
        let limit = extract_scalar_i64(&limit_tensor.data)?;
        let delta = extract_scalar_i64(&delta_tensor.data)?;

        if delta == 0 {
            return Err(OnnxError::ShapeInference(
                "Range delta cannot be zero".to_string(),
            ));
        }

        // Generate the range
        let mut values = Vec::new();
        if delta > 0 {
            let mut current = start;
            while current < limit {
                values.push(current);
                current += delta;
            }
        } else {
            let mut current = start;
            while current > limit {
                values.push(current);
                current += delta;
            }
        }

        let output = ConstantTensor {
            data: TensorData::Int64(values.clone()),
            shape: vec![values.len() as i64],
            data_type: TensorProto_DataType::INT64,
        };

        Ok(vec![output])
    }
}

// Helper function to extract a scalar i64 value from TensorData
fn extract_scalar_i64(data: &TensorData) -> Result<i64, OnnxError> {
    match data {
        TensorData::Int64(v) => {
            if v.is_empty() {
                Err(OnnxError::ShapeInference(
                    "Expected non-empty tensor".to_string(),
                ))
            } else {
                Ok(v[0])
            }
        }
        TensorData::Int32(v) => {
            if v.is_empty() {
                Err(OnnxError::ShapeInference(
                    "Expected non-empty tensor".to_string(),
                ))
            } else {
                Ok(v[0] as i64)
            }
        }
        _ => Err(OnnxError::ShapeInference(
            "Range only supports INT64 or INT32 inputs".to_string(),
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use onnx::onnx::TensorProto;
    use std::collections::HashMap;

    fn create_scalar_tensor(name: &str, value: i64) -> TensorProto {
        let mut tensor = TensorProto::new();
        tensor.set_name(name.to_string());
        tensor.set_data_type(TensorProto_DataType::INT64);
        tensor.set_dims(vec![]); // Scalar tensor
        tensor.set_raw_data(value.to_le_bytes().to_vec());
        tensor
    }

    #[test]
    fn test_range_evaluator_positive_delta() {
        let start = Box::leak(Box::new(create_scalar_tensor("start", 0)));
        let limit = Box::leak(Box::new(create_scalar_tensor("limit", 5)));
        let delta = Box::leak(Box::new(create_scalar_tensor("delta", 1)));

        let mut init_map = HashMap::new();
        init_map.insert("start".to_string(), start as &TensorProto);
        init_map.insert("limit".to_string(), limit as &TensorProto);
        init_map.insert("delta".to_string(), delta as &TensorProto);

        let ctx = ConstantFoldingContext::new(&init_map).unwrap();
        let evaluator = RangeEvaluator;

        let mut node = NodeProto::new();
        node.set_op_type("Range".to_string());
        node.set_input(protobuf::RepeatedField::from_vec(vec![
            "start".to_string(),
            "limit".to_string(),
            "delta".to_string(),
        ]));
        node.set_output(protobuf::RepeatedField::from_vec(
            vec!["output".to_string()],
        ));

        assert!(evaluator.can_evaluate(&node, &ctx));

        let result = evaluator.evaluate(&node, &ctx).unwrap();
        assert_eq!(result.len(), 1);

        let output = &result[0];
        assert_eq!(output.shape, vec![5]);
        assert_eq!(output.data_type, TensorProto_DataType::INT64);

        if let TensorData::Int64(ref values) = output.data {
            assert_eq!(values, &vec![0, 1, 2, 3, 4]);
        } else {
            panic!("Expected Int64 data");
        }
    }

    #[test]
    fn test_range_evaluator_step_2() {
        let start = Box::leak(Box::new(create_scalar_tensor("start", 0)));
        let limit = Box::leak(Box::new(create_scalar_tensor("limit", 10)));
        let delta = Box::leak(Box::new(create_scalar_tensor("delta", 2)));

        let mut init_map = HashMap::new();
        init_map.insert("start".to_string(), start as &TensorProto);
        init_map.insert("limit".to_string(), limit as &TensorProto);
        init_map.insert("delta".to_string(), delta as &TensorProto);

        let ctx = ConstantFoldingContext::new(&init_map).unwrap();
        let evaluator = RangeEvaluator;

        let mut node = NodeProto::new();
        node.set_op_type("Range".to_string());
        node.set_input(protobuf::RepeatedField::from_vec(vec![
            "start".to_string(),
            "limit".to_string(),
            "delta".to_string(),
        ]));
        node.set_output(protobuf::RepeatedField::from_vec(
            vec!["output".to_string()],
        ));

        let result = evaluator.evaluate(&node, &ctx).unwrap();
        assert_eq!(result.len(), 1);

        if let TensorData::Int64(ref values) = result[0].data {
            assert_eq!(values, &vec![0, 2, 4, 6, 8]);
        } else {
            panic!("Expected Int64 data");
        }
    }

    #[test]
    fn test_range_evaluator_negative_delta() {
        let start = Box::leak(Box::new(create_scalar_tensor("start", 5)));
        let limit = Box::leak(Box::new(create_scalar_tensor("limit", 0)));
        let delta = Box::leak(Box::new(create_scalar_tensor("delta", -1)));

        let mut init_map = HashMap::new();
        init_map.insert("start".to_string(), start as &TensorProto);
        init_map.insert("limit".to_string(), limit as &TensorProto);
        init_map.insert("delta".to_string(), delta as &TensorProto);

        let ctx = ConstantFoldingContext::new(&init_map).unwrap();
        let evaluator = RangeEvaluator;

        let mut node = NodeProto::new();
        node.set_op_type("Range".to_string());
        node.set_input(protobuf::RepeatedField::from_vec(vec![
            "start".to_string(),
            "limit".to_string(),
            "delta".to_string(),
        ]));
        node.set_output(protobuf::RepeatedField::from_vec(
            vec!["output".to_string()],
        ));

        let result = evaluator.evaluate(&node, &ctx).unwrap();

        if let TensorData::Int64(ref values) = result[0].data {
            assert_eq!(values, &vec![5, 4, 3, 2, 1]);
        } else {
            panic!("Expected Int64 data");
        }
    }
}
