use pasta_curves::arithmetic::{CurveAffine, Coordinates};
use std::ops::{Add, AddAssign};

pub trait CurveJacExt: CurveAffine
where
    Self::ExtendedJacobianCoordinates: Add<Self::ExtendedJacobianCoordinates, Output = Self::ExtendedJacobianCoordinates>
        + AddAssign<Self::ExtendedJacobianCoordinates>
        + Add<Self, Output = Self::ExtendedJacobianCoordinates>
        + AddAssign<Self>,
    Self::ExtendedJacobianCoordinates: From<<Self as CurveAffine>::CurveExt>
        + Add<<Self as CurveAffine>::CurveExt, Output = Self::ExtendedJacobianCoordinates>,
{
    type ExtendedJacobianCoordinates: Clone + Copy + Send + group::Group;

    fn jac_ext_identity() -> Self::ExtendedJacobianCoordinates {
        <Self::ExtendedJacobianCoordinates as group::Group>::identity()
    }

    fn jac_ext_to_affine(&self) -> Option<Coordinates<Self>> {
        self.coordinates().into()
    }

    // fn affine_to_jac_ext(&selff: CurveAffine) -> Self::ExtendedJacobianCoordinates {
    //     Self::jac_ext_identity() + *selff
    // }
}
