mod matrix;

use matrix::Matrix;

fn task1() {
    let mut mat = Matrix::<i8>::rand((3, 3));
    println!("A =\n{}", mat);
    mat.transpose_inplace();
    println!("A трансп. =\n{}", mat);
}

fn task2() {
    let mat1 = Matrix::<i8>::rand_range((3, 3), (-50, 50));
    let mat2 = Matrix::<i8>::rand_range((3, 3), (-50, 50));
    println!("A =\n{}", mat1);
    println!("B =\n{}", mat2);
    println!("A + B =\n{}", &mat1 + &mat2);
}

/*fn task3() {
    let mat1 = Matrix::<i16>::rand_range((3, 5), (-50, 50));
    let mat2 = Matrix::<i16>::rand_range((5, 2), (-50, 50));
    println!("A =\n{}", mat1);
    println!("B =\n{}", mat2);
    let product = matrix::mult_naive(&mat1, &mat2);
    println!("A * B =\n{}", product);
}

fn task4() {
    let mat1 = Matrix::<i16>::rand_range((3, 5), (-50, 50));
    let mat2 = Matrix::<i16>::rand_range((5, 2), (-50, 50));
    println!("A =\n{}", mat1);
    println!("B =\n{}", mat2);
    let product = matrix::mult_winograd(&mat1, &mat2);
    println!("A * B =\n{}", product);
}

fn task5() {
    let mat1 = Matrix::<i16>::rand_range((2, 2), (-50, 50));
    let mat2 = Matrix::<i16>::rand_range((2, 2), (-50, 50));
    println!("A =\n{}", mat1);
    println!("B =\n{}", mat2);
    let product = matrix::mult_strassen(&mat1, &mat2);
    println!("A * B =\n{}", product);
}*/

fn task_mult_all() {
    let mat1 = Matrix::<i16>::rand_range((2, 2), (-50, 50));
    let mat2 = Matrix::<i16>::rand_range((2, 2), (-50, 50));
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
    println!("Task 1");
    task1();
    println!("\nTask 2");
    task2();
    println!("\nTasks 3, 4, 5");
    task_mult_all();
}
