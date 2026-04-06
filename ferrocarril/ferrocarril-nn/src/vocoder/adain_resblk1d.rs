use crate::{
    Forward,
    adain::AdaIN1d,
    conv::Conv1d,
    conv_transpose::ConvTranspose1d,
};
use ferrocarril_core::tensor::Tensor;
use super::{UpSample1d, UpsampleType};

#[cfg(feature = "weights")]
use ferrocarril_core::FerroError;
#[cfg(feature = "weights")]
use ferrocarril_core::weights_binary::BinaryWeightLoader;
#[cfg(feature = "weights")]
use ferrocarril_core::LoadWeightsBinary;

pub struct AdainResBlk1d {
    upsample_enabled: bool,
    learned_sc: bool,
    dim_in: usize,
    dim_out: usize,
    norm1: AdaIN1d,
    norm2: AdaIN1d,
    conv1: Conv1d,
    conv2: Conv1d,
    pool: Option<ConvTranspose1d>,
    conv1x1: Option<Conv1d>,
    shortcut_upsample: Option<UpSample1d>,
}

impl AdainResBlk1d {
    pub fn new(
        dim_in: usize,
        dim_out: usize,
        style_dim: usize,
        upsample: bool,
        dropout_p: f32,
    ) -> Self {
        let _ = dropout_p;

        let learned_sc = dim_in != dim_out;

        let norm1 = AdaIN1d::new(style_dim, dim_in);
        let norm2 = AdaIN1d::new(style_dim, dim_out);
        let conv1 = Conv1d::new(dim_in, dim_out, 3, 1, 1, 1, 1, true);
        let conv2 = Conv1d::new(dim_out, dim_out, 3, 1, 1, 1, 1, true);

        let conv1x1 = if learned_sc {
            Some(Conv1d::new(dim_in, dim_out, 1, 1, 0, 1, 1, false))
        } else {
            None
        };

        let pool = if upsample {
            Some(ConvTranspose1d::new(
                dim_in,
                dim_in,
                3,
                2,
                1,
                1,
                dim_in,
                false,
            ))
        } else {
            None
        };

        let shortcut_upsample = if upsample {
            Some(UpSample1d::new(UpsampleType::Nearest))
        } else {
            None
        };

        Self {
            upsample_enabled: upsample,
            learned_sc,
            dim_in,
            dim_out,
            norm1,
            norm2,
            conv1,
            conv2,
            pool,
            conv1x1,
            shortcut_upsample,
        }
    }

    pub fn is_upsample(&self) -> bool {
        self.upsample_enabled
    }

    pub fn dim_in(&self) -> usize {
        self.dim_in
    }

    pub fn dim_out(&self) -> usize {
        self.dim_out
    }

    fn leaky_relu(x: &Tensor<f32>) -> Tensor<f32> {
        let mut data = x.data().to_vec();
        for v in data.iter_mut() {
            if *v < 0.0 {
                *v *= 0.2;
            }
        }
        Tensor::from_data(data, x.shape().to_vec())
    }

    fn residual(&self, x: &Tensor<f32>, s: &Tensor<f32>) -> Tensor<f32> {
        let mut y = self.norm1.forward(x, s);
        y = Self::leaky_relu(&y);

        if let Some(ref pool) = self.pool {
            y = pool.forward(&y);
        }

        y = self.conv1.forward(&y);
        y = self.norm2.forward(&y, s);
        y = Self::leaky_relu(&y);
        y = self.conv2.forward(&y);

        y
    }

    fn shortcut(&self, x: &Tensor<f32>) -> Tensor<f32> {
        let mut y = x.clone();

        if let Some(ref up) = self.shortcut_upsample {
            y = up.forward(&y);
        }

        if let Some(ref c) = self.conv1x1 {
            y = c.forward(&y);
        }

        y
    }

    pub fn forward(&self, x: &Tensor<f32>, s: &Tensor<f32>) -> Tensor<f32> {
        let res = self.residual(x, s);
        let sc = self.shortcut(x);

        assert_eq!(
            res.shape(),
            sc.shape(),
            "AdainResBlk1d: residual shape {:?} != shortcut shape {:?} (dim_in={}, dim_out={}, upsample={})",
            res.shape(),
            sc.shape(),
            self.dim_in,
            self.dim_out,
            self.upsample_enabled
        );

        let scale = 1.0_f32 / (2.0_f32).sqrt();
        let mut out_data = Vec::with_capacity(res.data().len());

        for (&a, &b) in res.data().iter().zip(sc.data().iter()) {
            out_data.push((a + b) * scale);
        }

        Tensor::from_data(out_data, res.shape().to_vec())
    }
}

#[cfg(feature = "weights")]
impl LoadWeightsBinary for AdainResBlk1d {
    fn load_weights_binary(
        &mut self,
        loader: &BinaryWeightLoader,
        component: &str,
        prefix: &str,
    ) -> Result<(), FerroError> {
        self.norm1
            .load_weights_binary(loader, component, &format!("{}.norm1", prefix))?;
        self.norm2
            .load_weights_binary(loader, component, &format!("{}.norm2", prefix))?;
        self.conv1
            .load_weights_binary(loader, component, &format!("{}.conv1", prefix))?;
        self.conv2
            .load_weights_binary(loader, component, &format!("{}.conv2", prefix))?;

        if self.learned_sc && self.conv1x1.is_some() {
            self.conv1x1
                .as_mut()
                .unwrap()
                .load_weights_binary(loader, component, &format!("{}.conv1x1", prefix))?;
        }

        if self.upsample_enabled && self.pool.is_some() {
            self.pool
                .as_mut()
                .unwrap()
                .load_weights_binary(loader, component, &format!("{}.pool", prefix))?;
        }

        Ok(())
    }
}