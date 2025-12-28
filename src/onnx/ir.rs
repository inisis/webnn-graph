// Minimal ONNX IR structures used for static shape/type inference.
use crate::ast::DataType;
use std::collections::HashMap;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Dim {
    Known(i64),
    Unknown(Option<String>),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorShape {
    pub dims: Vec<Dim>,
}

impl TensorShape {
    pub fn from_known(dims: Vec<i64>) -> Self {
        Self {
            dims: dims.into_iter().map(Dim::Known).collect(),
        }
    }

    pub fn is_static(&self) -> bool {
        self.dims.iter().all(|d| matches!(d, Dim::Known(_)))
    }

    pub fn to_i64(&self) -> Option<Vec<i64>> {
        self.dims
            .iter()
            .map(|d| match d {
                Dim::Known(v) => Some(*v),
                Dim::Unknown(_) => None,
            })
            .collect()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorType {
    pub data_type: DataType,
    pub shape: TensorShape,
}

#[derive(Debug, Clone)]
pub struct ValueInfo {
    pub id: String,
    pub ty: Option<TensorType>,
    pub producer: Option<String>,
    pub consumers: Vec<String>,
}

impl ValueInfo {
    pub fn new(id: String) -> Self {
        Self {
            id,
            ty: None,
            producer: None,
            consumers: Vec::new(),
        }
    }
}

#[derive(Debug, Default, Clone)]
pub struct OnnxIrGraph {
    pub values: HashMap<String, ValueInfo>,
}

impl OnnxIrGraph {
    pub fn value_or_insert(&mut self, id: &str) -> &mut ValueInfo {
        self.values
            .entry(id.to_string())
            .or_insert_with(|| ValueInfo::new(id.to_string()))
    }
}
