// Normalization operators: LayerNormalization, Softmax

use crate::ast::Node;
use crate::onnx::convert::{sanitize_identifier, OnnxError};
use crate::onnx::ops::{ConversionContext, ConversionResult, OpHandler};
use onnx::onnx::NodeProto;
use serde_json::Map;

pub struct NormalizationHandler;

impl OpHandler for NormalizationHandler {
    fn supports(&self, op_type: &str) -> bool {
        matches!(op_type, "LayerNormalization" | "Softmax")
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
            "LayerNormalization" => self.convert_layer_norm(node, &node_name, context),
            "Softmax" => self.convert_softmax(node, &node_name, context),
            _ => Err(OnnxError::UnsupportedOp {
                op: op_type.to_string(),
                node: node_name,
            }),
        }
    }
}

impl NormalizationHandler {
    /// Convert ONNX LayerNormalization to WebNN layerNormalization
    fn convert_layer_norm(
        &self,
        node: &NodeProto,
        node_name: &str,
        context: &ConversionContext,
    ) -> Result<ConversionResult, OnnxError> {
        let inputs = node.get_input();
        if inputs.is_empty() {
            return Err(OnnxError::InvalidShape(
                "LayerNormalization expects at least 1 input".to_string(),
            ));
        }

        // Extract attributes
        let mut epsilon = 1e-5f32;
        let mut axis = -1i64;

        for attr in node.get_attribute() {
            match attr.get_name() {
                "epsilon" => {
                    if attr.has_f() {
                        epsilon = attr.get_f();
                    }
                }
                "axis" => {
                    if attr.has_i() {
                        axis = attr.get_i();
                    }
                }
                _ => {}
            }
        }

        let output_name = if node.get_output().is_empty() {
            format!("{}_output", node_name)
        } else {
            sanitize_identifier(&node.get_output()[0].to_string())
        };

        let mut options = Map::new();
        options.insert("epsilon".to_string(), serde_json::json!(epsilon));

        // WebNN layerNormalization uses axes parameter (array)
        // Convert ONNX axis to axes array
        if axis != -1 {
            options.insert("axes".to_string(), serde_json::json!([axis]));
        }

        // LayerNormalization can have scale and bias as inputs
        let webnn_inputs = if inputs.len() >= 3 {
            // Input, scale, bias
            let input0 = context.resolve_input(&inputs[0]);
            let input1 = context.resolve_input(&inputs[1]);
            let input2 = context.resolve_input(&inputs[2]);
            vec![input0, input1, input2]
        } else if inputs.len() == 2 {
            // Input, scale
            let input0 = context.resolve_input(&inputs[0]);
            let input1 = context.resolve_input(&inputs[1]);
            vec![input0, input1]
        } else {
            // Just input
            let input0 = context.resolve_input(&inputs[0]);
            vec![input0]
        };

        let mut result = ConversionResult::new(vec![Node {
            id: output_name.clone(),
            op: "layerNormalization".to_string(),
            inputs: webnn_inputs,
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

    /// Convert ONNX Softmax to WebNN softmax
    fn convert_softmax(
        &self,
        node: &NodeProto,
        node_name: &str,
        context: &ConversionContext,
    ) -> Result<ConversionResult, OnnxError> {
        let inputs = node.get_input();
        if inputs.len() != 1 {
            return Err(OnnxError::InvalidShape(format!(
                "Softmax expects 1 input, got {}",
                inputs.len()
            )));
        }

        // Extract axis attribute
        let mut axis = -1i64;
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

        let mut options = Map::new();
        // WebNN softmax uses axis parameter (single value)
        options.insert("axis".to_string(), serde_json::json!(axis));

        let mut result = ConversionResult::new(vec![Node {
            id: output_name.clone(),
            op: "softmax".to_string(),
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
    fn test_normalization_handler_supports() {
        let handler = NormalizationHandler;
        assert!(handler.supports("LayerNormalization"));
        assert!(handler.supports("Softmax"));
        assert!(!handler.supports("Add"));
    }

    #[test]
    fn test_convert_softmax() {
        let handler = NormalizationHandler;
        let mut node = create_test_node("Softmax", vec!["x"], vec!["y"]);
        add_int_attribute(&mut node, "axis", -1);
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
        assert_eq!(result.nodes[0].op, "softmax");
        assert_eq!(result.nodes[0].inputs, vec!["x"]);
        assert_eq!(result.nodes[0].id, "y");
        assert!(result.nodes[0].options.contains_key("axis"));
    }

    #[test]
    fn test_convert_layer_norm() {
        let handler = NormalizationHandler;
        let node = create_test_node("LayerNormalization", vec!["x", "scale", "bias"], vec!["y"]);
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
        assert_eq!(result.nodes[0].op, "layerNormalization");
        assert_eq!(result.nodes[0].inputs.len(), 3);
        assert!(result.nodes[0].options.contains_key("epsilon"));
    }
}
