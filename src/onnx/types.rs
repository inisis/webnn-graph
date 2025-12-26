// ONNX to WebNN type mapping

use crate::ast::DataType;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum TypeConversionError {
    #[error("unsupported ONNX data type: {0}")]
    UnsupportedDataType(i32),
}

/// Map ONNX TensorProto data type to WebNN DataType
pub fn map_onnx_data_type(onnx_type: i32) -> Result<DataType, TypeConversionError> {
    match onnx_type {
        1 => Ok(DataType::Float32),  // FLOAT
        10 => Ok(DataType::Float16), // FLOAT16
        6 => Ok(DataType::Int32),    // INT32
        12 => Ok(DataType::Uint32),  // UINT32
        7 => Ok(DataType::Int64),    // INT64
        13 => Ok(DataType::Uint64),  // UINT64
        3 => Ok(DataType::Int8),     // INT8
        2 => Ok(DataType::Uint8),    // UINT8
        _ => Err(TypeConversionError::UnsupportedDataType(onnx_type)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_data_type_mapping() {
        assert_eq!(map_onnx_data_type(1).unwrap(), DataType::Float32);
        assert_eq!(map_onnx_data_type(10).unwrap(), DataType::Float16);
        assert_eq!(map_onnx_data_type(6).unwrap(), DataType::Int32);
        assert_eq!(map_onnx_data_type(12).unwrap(), DataType::Uint32);
        assert_eq!(map_onnx_data_type(7).unwrap(), DataType::Int64);
        assert_eq!(map_onnx_data_type(13).unwrap(), DataType::Uint64);
        assert_eq!(map_onnx_data_type(3).unwrap(), DataType::Int8);
        assert_eq!(map_onnx_data_type(2).unwrap(), DataType::Uint8);
    }

    #[test]
    fn test_unsupported_type() {
        assert!(map_onnx_data_type(999).is_err());
    }
}
