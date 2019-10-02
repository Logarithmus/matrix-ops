mod matrix;

use matrix::Matrix;

fn task1() {
    println!("Task 1: in-place matrix tranpose");
    let mut mat = Matrix::<f32>::rand_range((3, 3), (-50.0, 50.0));
    println!("A =\n{}", mat);
    mat.transpose_inplace();
    println!("A transposed =\n{}", mat);
}

fn task2() {
    println!("Task 2: matrix sum");
    let mat1 = Matrix::<f32>::rand_range((3, 3), (-50.0, 50.0));
    let mat2 = Matrix::<f32>::rand_range((3, 3), (-50.0, 50.0));
    println!("A =\n{}", mat1);
    println!("B =\n{}", mat2);
    println!("A + B =\n{}", &mat1 + &mat2);
}

fn tasks_3_4_5() {
    println!("Tasks 3, 4, 5: matrix multiplication");
    let mat1 = Matrix::<f32>::rand_range((2, 5), (-50.0, 50.0));
    let mat2 = Matrix::<f32>::rand_range((5, 3), (-50.0, 50.0));
    println!("A =\n{}", mat1);
    println!("B =\n{}", mat2);
    let product = matrix::mult_naive(&mat1, &mat2);
    println!("Naive A * B =\n{}", product);
    let product = matrix::mult_winograd(&mat1, &mat2);
    println!("Winograd A * B =\n{}", product);
    let product = matrix::mult_strassen(&mat1, &mat2);
    println!("Strassen A * B =\n{}", product);
}

fn main() {
    task1();
    println!("----------------------------");
    task2();
    println!("----------------------------");
    tasks_3_4_5();
}
