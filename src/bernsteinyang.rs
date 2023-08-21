use core::cmp::PartialEq;
use std::ops::{Add, Mul, Sub, Neg};

/// Big signed (B * L)-bit integer type, whose variables store
/// numbers in the two's complement code as arrays of B-bit chunks.
/// The ordering of the chunks in these arrays is little-endian.
/// The arithmetic operations for this type are wrapping ones.
#[derive(Clone)]
struct ChunkInt<const B:usize, const L:usize>(pub [u64; L]);

impl<const B:usize, const L:usize> ChunkInt<B,L> {
    /// Mask, in which the B lowest bits are 1 and only they
    pub const MASK: u64 = u64::MAX >> (64 - B);

    /// Representation of 0
    pub const ZERO: Self = Self([0; L]);
    
    /// Representation of 1
    pub const ONE: Self = { 
        let mut data = [0; L]; 
        data[0] = 1; Self(data) 
    };

    /// Representation of -1
    pub const MINUS_ONE: Self = Self([Self::MASK; L]);

    /// Returns the result of applying B-bit right
    /// arithmetical shift to the current number
    pub fn shift(&self) -> Self {
        let mut data = [0; L];
        for i in 1..L {
            data[i - 1] = self.0[i];
        }
        if self.is_negative() {
            data[L - 1] = Self::MASK;
        }
        Self(data)
    }

    /// Returns the lowest B bits of the current number
    pub fn lowest(&self) -> u64 { self.0[0] }

    /// Returns "true" iff the current number is negative
    pub fn is_negative(&self) -> bool {
        self.0[L - 1] > (Self::MASK >> 1)
    }
}

impl <const B:usize, const L:usize> PartialEq for ChunkInt<B,L> {
    fn eq(&self, other: &Self) -> bool { self.0 == other.0 }
    fn ne(&self, other: &Self) -> bool { self.0 != other.0 } 
}

impl<const B:usize, const L:usize> Add for &ChunkInt<B,L> {
    type Output = ChunkInt<B,L>;
    fn add(self, other: Self) -> Self::Output {
        let (mut data, mut carry) = ([0; L], 0);
        for i in 0..L {
            let sum = self.0[i] + other.0[i] + carry;
            data[i] = sum & ChunkInt::<B,L>::MASK;
            carry = sum >> B;
        }
        Self::Output { 0: data }
    }
}

impl<const B:usize, const L:usize> Add<&ChunkInt<B,L>> for ChunkInt<B,L> {
    type Output = ChunkInt<B,L>;
    fn add(self, other: &Self) -> Self::Output {
        &self + other
    }
}

impl<const B:usize, const L:usize> Add for ChunkInt<B,L> {
    type Output = ChunkInt<B,L>;
    fn add(self, other: Self) -> Self::Output {
        &self + &other
    }
}

impl<const B:usize, const L:usize> Sub for &ChunkInt<B,L> {
    type Output = ChunkInt<B,L>;
    fn sub(self, other: Self) -> Self::Output {
        let (mut data, mut carry) = ([0; L], 1);
        for i in 0..L {
            let sum = self.0[i] + (other.0[i] ^ ChunkInt::<B,L>::MASK) + carry;
            data[i] = sum & ChunkInt::<B,L>::MASK;
            carry = sum >> B;
        }
        Self::Output { 0: data }
    }
}

impl<const B:usize, const L:usize> Sub<&ChunkInt<B,L>> for ChunkInt<B,L> {
    type Output = ChunkInt<B,L>;
    fn sub(self, other: &Self) -> Self::Output {
        &self - other
    }
}

impl<const B:usize, const L:usize> Sub for ChunkInt<B,L> {
    type Output = ChunkInt<B,L>;
    fn sub(self, other: Self) -> Self::Output {
        &self - &other
    }
}

impl<const B:usize, const L:usize> Neg for &ChunkInt<B,L> {
    type Output = ChunkInt<B,L>;
    fn neg(self) -> Self::Output {
        let (mut data, mut carry) = ([0; L], 1);
        for i in 0..L {
            let sum = (self.0[i] ^ ChunkInt::<B,L>::MASK) + carry;
            data[i] = sum & ChunkInt::<B,L>::MASK;
            carry = sum >> B;
        }
        Self::Output { 0: data }
    }
}

impl<const B:usize, const L:usize> Neg for ChunkInt<B,L> {
    type Output = ChunkInt<B,L>;
    fn neg(self) -> Self::Output {
        -&self
    }
}

impl<const B:usize, const L:usize> Mul for &ChunkInt<B,L> {
    type Output = ChunkInt<B,L>;
    fn mul(self, other: Self) -> Self::Output {
        let mut data = [0; L];
        for i in 0..L {
            let mut carry = 0;
            for k in 0..(L - i) {
                let sum = (data[i + k] as u128) + (carry as u128) +
                    (self.0[i] as u128) * (other.0[k] as u128);
                data[i + k] = sum as u64 & ChunkInt::<B,L>::MASK;
                carry = (sum >> B) as u64;
            }
        }
        Self::Output { 0: data }
    }
}

impl<const B:usize, const L:usize> Mul<&ChunkInt<B,L>> for ChunkInt<B,L> {
    type Output = ChunkInt<B,L>;
    fn mul(self, other: &Self) -> Self::Output {
        &self * other
    }
}

impl<const B:usize, const L:usize> Mul for ChunkInt<B,L> {
    type Output = ChunkInt<B,L>;
    fn mul(self, other: Self) -> Self::Output {
        &self * &other
    }
}

impl<const B:usize, const L:usize> Mul<i64> for &ChunkInt<B,L> {
    type Output = ChunkInt<B,L>;
    fn mul(self, other: i64) -> Self::Output {
        let mut data = [0; L];
        let (other, mut carry, mask) = if other < 0 { 
            (-other, -other as u64, ChunkInt::<B,L>::MASK) 
        } else { (other, 0, 0) };
        for i in 0..L {
            let sum = (carry as u128) + ((self.0[i] ^ mask) as u128) * (other as u128);
            data[i] = sum as u64 & ChunkInt::<B,L>::MASK;
            carry = (sum >> B) as u64;
        }
        Self::Output { 0: data }
    }
}

impl<const B:usize, const L:usize> Mul<i64> for ChunkInt<B,L> {
    type Output = ChunkInt<B,L>;
    fn mul(self, other: i64) -> Self::Output {
        &self * other
    }
}

impl<const B:usize, const L:usize> Mul<&ChunkInt<B,L>> for i64 {
    type Output = ChunkInt<B,L>;
    fn mul(self, other: &ChunkInt<B,L>) -> Self::Output {
        other * self 
    }
}

impl<const B:usize, const L:usize> Mul<ChunkInt<B,L>> for i64 {
    type Output = ChunkInt<B,L>;
    fn mul(self, other: ChunkInt<B,L>) -> Self::Output {
        other * self 
    }
}

/// Type of the modular multiplicative inverter based on the Bernstein-Yang method.
/// The inverter can be created for a specified modulus M and adjusting parameter A 
/// to compute the adjusted multiplicative inverses of positive integers, i.e. for 
/// computing (1 / x) * A mod M for a positive integer x.
///
/// The adjusting parameter allows computing the multiplicative inverses in the case of
/// using the Montgomery representation for the input or the expected output. If R is 
/// the Montgomery factor, the multiplicative inverses in the appropriate representation
/// can be computed provided that the value of A is chosen as follows:
/// - A = 1, if both the input and the expected output are in the trivial form
/// - A = R^2 mod M, if both the input and the expected output are in the Montgomery form
/// - A = R mod M, if either the input or the expected output is in the Montgomery form,
/// but not both of them
/// 
/// The public methods of this type receive and return unsigned big integers as arrays of
/// 64-bit chunks, the ordering of which is little-endian. Both the modulus and the integer
/// to be inverted should not exceed 2 ^ (B * (L - 1) - 2)
pub struct BYInverter<const B:usize, const L:usize> {
    /// Modulus
    modulus: ChunkInt<B,L>,

    /// Adjusting parameter
    adjuster: ChunkInt<B,L>,

    /// Multiplicative inverse of the modulus modulo 2^B
    inverse: i64
}

/// Type of the Bernstein-Yang transition matrix multiplied by 2^B
type Matrix = [[i64; 2]; 2];

impl<const B:usize, const L:usize> BYInverter<B,L> {  
    fn step(f: &ChunkInt<B,L>, g: &ChunkInt<B,L>, mut delta: i64) -> (i64, Matrix) {
        let (mut steps, mut f, mut g) = (B as i64, f.lowest() as i64, g.lowest() as i128);
        let mut matrix: Matrix = [[1, 0], [0, 1]];

        loop {
            let zeros = steps.min(g.trailing_zeros() as i64);
            (steps, delta, g) = (steps - zeros, delta + zeros, g >> zeros);
            matrix[0] = [matrix[0][0] << zeros, matrix[0][1] << zeros];
    
            if steps == 0  { break; }      
            
            if delta > 0 {
                (delta, f, g) =  (-delta, g as i64, -f as i128);
                (matrix[0], matrix[1]) =  (matrix[1], [-matrix[0][0], -matrix[0][1]]);
            }

            let mask = (1 << steps.min(1 - delta).min(4)) - 1;
            let w = (g as i64).wrapping_mul(f.wrapping_mul(3) ^ 12) & mask;
            
            matrix[1] = [matrix[0][0] * w + matrix[1][0], matrix[0][1] * w + matrix[1][1]];
            g += w as i128 * f as i128;
        }
    
        (delta, matrix)
    }
    
    fn fg(f: ChunkInt<B,L>, g: ChunkInt<B,L>, matrix: Matrix) -> (ChunkInt<B,L>, ChunkInt<B,L>) {
        ((matrix[0][0] * &f + matrix[0][1] * &g).shift(), (matrix[1][0] * &f + matrix[1][1] * &g).shift()) 
    }

    fn de(&self, d: ChunkInt<B,L>, e: ChunkInt<B,L>, matrix: Matrix) -> (ChunkInt<B,L>, ChunkInt<B,L>) {
        let mask = ChunkInt::<B,L>::MASK as i64;
        let mut md = matrix[0][0] * d.is_negative() as i64 + matrix[0][1] * e.is_negative() as i64;
        let mut me = matrix[1][0] * d.is_negative() as i64 + matrix[1][1] * e.is_negative() as i64;

        let cd = matrix[0][0].wrapping_mul(d.lowest() as i64).wrapping_add(matrix[0][1].wrapping_mul(e.lowest() as i64)) & mask; 
        let ce = matrix[1][0].wrapping_mul(d.lowest() as i64).wrapping_add(matrix[1][1].wrapping_mul(e.lowest() as i64)) & mask;

        md -= (self.inverse.wrapping_mul(cd).wrapping_add(md)) & mask;
        me -= (self.inverse.wrapping_mul(ce).wrapping_add(me)) & mask;
        
        let cd = matrix[0][0] * &d + matrix[0][1] * &e + md * &self.modulus;
        let ce = matrix[1][0] * &d + matrix[1][1] * &e + me * &self.modulus;

        (cd.shift(), ce.shift())   
    }

    fn norm(&self, mut value: ChunkInt<B,L>, negate: bool) -> ChunkInt<B,L> {
        if value.is_negative() {
            value = value + &self.modulus;
        }

        if negate {
            value = -value;
        }

        if value.is_negative() {
            value = value + &self.modulus;
        }

        value
    }

    /// Returns a big unsigned integer as an array of O-bit chunks, which is equal modulo
    /// 2 ^ (O * S) to the input big unsigned integer stored as an array of I-bit chunks.
    /// The ordering of the chunks in these arrays is little-endian
    const fn convert<const I:usize, const O:usize, const S:usize>(input: &[u64]) -> [u64; S] {
        const fn min(a: usize, b: usize) -> usize { if a > b { b } else { a } }
        let (total, mut output, mut bits) = (min(input.len() * I, S * O), [0; S], 0);
          
        while bits < total {
            let (i, o) = (bits % I, bits % O);
            output[bits / O] |= (input[bits / I] >> i) << o;
            bits += min(I - i, O - o);
        }
        
        let mask = u64::MAX >> (64 - O);
        let mut filled = total / O + if total % O > 0 { 1 } else { 0 };
    
        while filled > 0 {
            filled -= 1;
            output[filled] &= mask;
        }
        
        output
    }

    /// Returns the multiplicative inverse of the argument modulo 2^B. The implementation is based
    /// on the Hurchalla's method for computing the multiplicative inverse modulo a power of two
    const fn inv(value: u64) -> i64 {
        let x = value.wrapping_mul(3) ^ 2;
        let y = 1u64.wrapping_sub(x.wrapping_mul(value));
        let (x, y) = (x.wrapping_mul(y.wrapping_add(1)), y.wrapping_mul(y));
        let (x, y) = (x.wrapping_mul(y.wrapping_add(1)), y.wrapping_mul(y));
        let (x, y) = (x.wrapping_mul(y.wrapping_add(1)), y.wrapping_mul(y));
        (x.wrapping_mul(y.wrapping_add(1)) & ChunkInt::<B,L>::MASK) as i64
    }

    /// Creates the inverter for specified modulus and adjusting parameter
    pub const fn new(modulus: &[u64], adjuster: &[u64]) -> Self {
        Self {
            modulus: ChunkInt::<B, L>(Self::convert::<64, B, L>(modulus)), 
            adjuster: ChunkInt::<B, L>(Self::convert::<64, B, L>(adjuster)),
            inverse: Self::inv(modulus[0])
        }
    }
    
    /// Returns either the adjusted modular multiplicative inverse for the argument or None 
    /// depending on invertibility of the argument, i.e. its coprimality with the modulus
    pub fn invert<const S:usize>(&self, value: &[u64]) -> Option<[u64; S]> {
        let (mut d, mut e) = (ChunkInt::ZERO, self.adjuster.clone());
        let mut g = ChunkInt::<B, L>(Self::convert::<64, B, L>(value)); 
        let (mut delta, mut f) = (1, self.modulus.clone());
        let mut matrix;
        while g != ChunkInt::ZERO {
            (delta, matrix) = Self::step(&f, &g, delta);
            (f, g) = Self::fg(f, g, matrix);
            (d, e) = self.de(d, e, matrix);
        }
        let antiunit = f == ChunkInt::MINUS_ONE;
        if (f != ChunkInt::ONE) && !antiunit { return None; }
        Some(Self::convert::<B, 64, S>(&self.norm(d, antiunit).0))
    }
}