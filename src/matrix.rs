use core::ops::{Add, Sub, AddAssign, SubAssign, Index, IndexMut, Mul};
use rand::distributions::{uniform::SampleUniform, Distribution, Standard};
use rand::Rng;
use std::default::Default;
use std::fmt::{self, Debug, Display, Formatter};
use std::iter::Sum;

pub struct Matrix<T> {
    data: Vec<T>,
    rows: usize,
    cols: usize,
}

impl<T> Index<(usize, usize)> for Matrix<T> {
    type Output = T;
    fn index(&self, index: (usize, usize)) -> &Self::Output {
        let (row, col) = index;
        &self.data[self.cols * row + col]
    }
}

impl<T> IndexMut<(usize, usize)> for Matrix<T> {
    fn index_mut(&mut self, index: (usize, usize)) -> &mut T {
        let (row, col) = index;
        &mut self.data[self.cols * row + col]
    }
}


impl<'a, T: Add<Output = T> + Copy> Add for &'a Matrix<T> {
    type Output = Matrix<T>;
    fn add(self, other: Self) -> Self::Output {
        let iter12 = self.data.iter().zip(other.data.iter());
        Matrix::with_data(iter12.map(|(&a, &b)| a + b).collect(), (self.rows, self.cols)) 
    }
}

impl<'a, T: Sub<Output = T> + Copy> Sub for &'a Matrix<T> {
    type Output = Matrix<T>;
    fn sub(self, other: Self) -> Self::Output {
        let iter12 = self.data.iter().zip(other.data.iter());
        Matrix::with_data(iter12.map(|(&a, &b)| a - b).collect(), (self.rows, self.cols)) 
    }
}

impl<T> Display for Matrix<T>
where
    T: Debug,
{
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        for row in self.data.chunks(self.cols) {
            write!(f, "{:<5?}", row)?;
            println!();
        }
        Ok(())
    }
}

impl<T> Matrix<T> {
    pub fn new((rows, cols): (usize, usize)) -> Matrix<T>
    where
        T: Clone + Default,
    {
        let data = vec![T::default(); rows * cols];
        Matrix { data, rows, cols }
    }

    pub fn with_data(data: Vec<T>, (rows, cols): (usize, usize)) -> Matrix<T> {
        Matrix { data, rows, cols }
    }

    pub fn from_quadrants(q: &[[Matrix<T>; 2]; 2]) -> Matrix<T>
    where T: Copy
    {
        let (rows, cols) = (q[0][0].rows, q[0][0].cols);
        let row_iters = 
        (
            q[0][0].data.chunks(cols),
            q[0][1].data.chunks(cols),
            q[1][0].data.chunks(cols),
            q[1][1].data.chunks(cols),
        );
        let q0_iter = row_iters.0.zip(row_iters.1).map(|row| row.0.iter().chain(row.1.iter())).flatten();
        let q1_iter = row_iters.2.zip(row_iters.3).map(|row| row.0.iter().chain(row.1.iter())).flatten();
        let data: Vec<T> = q0_iter.chain(q1_iter).copied().collect();
        Matrix::with_data(data, (rows * 2, cols * 2)) 
    }

    pub fn rand((rows, cols): (usize, usize)) -> Matrix<T>
    where
        Standard: Distribution<T>,
    {
        let mut rng = rand::thread_rng();
        let data = (0..rows * cols).map(|_| rng.gen()).collect();
        Matrix { data, rows, cols }
    }

    pub fn rand_range((rows, cols): (usize, usize), (low, high): (T, T)) -> Matrix<T>
    where
        T: SampleUniform + Copy,
    {
        let mut rng = rand::thread_rng();
        let data = (0..rows * cols).map(|_| rng.gen_range(low, high)).collect();
        Matrix { data, rows, cols }
    }

    pub fn transpose_inplace(&mut self) {
        assert_eq!(self.rows, self.cols);
        for i in 0..(self.rows - 1) {
            for j in (i + 1)..self.cols {
                self.data.swap(self.rows * i + j, self.rows * j + i);
            }
        }
    }

    pub fn quadrants(&self) -> [[Matrix<T>; 2]; 2]
    where T: Copy {
        let (rows, cols, data) = (self.rows, self.cols, &self.data);
        let top_left = data.chunks(cols).take(rows / 2)
            .map(|row| &row[..(cols / 2)])
            .flatten();
        let top_right = data.chunks(cols).take(rows / 2)
            .map(|row| &row[(cols / 2)..])
            .flatten();
        let bottom_left = data.chunks(cols).skip(rows / 2)
            .map(|row| &row[..(cols / 2)])
            .flatten();
        let bottom_right = data.chunks(cols).skip(rows / 2)
            .map(|row| &row[(cols / 2)..])
            .flatten();
        [
            [
                Matrix::with_data(top_left.copied().collect(), (rows / 2, cols / 2)),
                Matrix::with_data(top_right.copied().collect(), (rows / 2, cols / 2)),
            ],
            [
                Matrix::with_data(bottom_left.copied().collect(), (rows / 2, cols / 2)),
                Matrix::with_data(bottom_right.copied().collect(), (rows / 2, cols / 2)),
            ],
        ]
    }
}

pub fn mult_naive<T>(mat1: &Matrix<T>, mat2: &Matrix<T>) -> Matrix<T>
where
    T: Mul<Output = T> + Add<Output = T> + AddAssign + Default + Clone + Copy + Display,
{
    assert_eq!(mat1.cols, mat2.rows);
    let mut product = Matrix::new((mat1.rows, mat2.cols));
    for row in 0..mat1.rows {
        for k in 0..mat1.cols {
            for col in 0..mat2.cols {
                product[(row, col)] += mat1[(row, k)] * mat2[(k, col)];
            }
        }
    }
    product
}

pub fn mult_winograd<T: Display>(mat1: &Matrix<T>, mat2: &Matrix<T>) -> Matrix<T>
where
    T: Mul<Output = T> + Add<Output = T> + AddAssign + SubAssign + Sum + Default + Copy,
{
    let rows_pairs_iter = mat1.data.chunks(mat1.cols).map(|row| row.chunks_exact(2));
    let row_factors: Vec<T> = rows_pairs_iter
        .map(|pairs| pairs.map(|pair| pair[0] * pair[1]).sum())
        .collect();

    let mut col_factors = Vec::with_capacity(mat2.cols);
    for j in 0..mat2.cols {
        col_factors.push((0..(mat2.rows / 2) * 2).step_by(2)
                .map(|i| mat2[(i, j)] * mat2[(i + 1, j)])
                .sum(),
        );
    }

    let mut product = Matrix::new((mat1.rows, mat2.cols));
    for row in 0..product.rows {
        for k in (0..(mat1.cols / 2) * 2).step_by(2) {
            for col in 0..product.cols {
                product[(row, col)] +=
                    (mat1[(row, k)] + mat2[(k + 1, col)]) * (mat1[(row, k + 1)] + mat2[(k, col)]);
            }
        }
    }

    for row in 0..product.rows {
        for col in 0..product.cols {
            product[(row, col)] -= row_factors[row] + col_factors[col];
        }
    }
    if mat1.cols % 2 == 1 {
        for row in 0..product.rows {
            for col in 0..product.cols {
                product[(row, col)] += mat1[(row, mat1.cols - 1)] * mat2[(mat2.rows - 1, col)];
            }
        }
    }
    product
}

pub fn mult_strassen<T>(mat1: &Matrix<T>, mat2: &Matrix<T>) -> Matrix<T>
where
    T: Mul<Output = T> + Add<Output = T> + Sub<Output = T> + Copy,
{
    if (mat1.rows == 1) && (mat1.cols == 1) {
        return Matrix::with_data(vec![mat1[(0, 0)] * mat2[(0, 0)]; 1], (1, 1));
    }
    let (a, b) = (mat1.quadrants(), mat2.quadrants());
    let m = [
                mult_strassen(&(&a[0][0] + &a[1][1]), &(&b[0][0] + &b[1][1])),
                mult_strassen(&(&a[1][0] + &a[1][1]), &b[0][0]),
                mult_strassen(&a[0][0], &(&b[0][1] - &b[1][1])),
                mult_strassen(&a[1][1], &(&b[1][0] - &b[0][0])),
                mult_strassen(&(&a[0][0] + &a[0][1]), &b[1][1]),
                mult_strassen(&(&a[1][0] - &a[0][0]), &(&b[0][0] + &b[0][1])),
                mult_strassen(&(&a[0][1] - &a[1][1]), &(&b[1][0] + &b[1][1]))
            ];
    let c = [
                [
                    &(&(&m[0] + &m[3]) - &m[4]) + &m[6],
                    &m[2] + &m[4],
                ],
                [
                    &m[1] + &m[3],
                    &(&(&m[0] - &m[1]) + &m[2]) + &m[5],
                ]
            ];
    Matrix::from_quadrants(&c)
}