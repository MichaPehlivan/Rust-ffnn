use nalgebra::SMatrix;
use nalgebra::SVector;
use std::f64::consts::E;
use rand::random;

type Matrix4x3 = SMatrix<f64, 4, 3>;
type Matrix2x2 = SMatrix<f64, 2, 2>;
type Vector1x2 = SMatrix<f64, 1, 2>;
type Vector2x1 = SVector<f64, 2>;
type Vector1x1 = SVector<f64, 1>;

//neural network with 2 input, 2 hidden, and 1 output neuron(s)
struct Network {
    w1: Matrix2x2,
    b1: Vector1x2,
    w2: Vector2x1,
    b2: Vector1x1,
}

//initialize network with random weights and biases 
fn init() -> Network {
    Network {
        w1: Matrix2x2::new(random(), random(),
                           random(), random()),
        b1: Vector1x2::new(random(), random()),
        w2: Vector2x1::new(random(), random()),
        b2: Vector1x1::new(random()),
    }
}

//feeds input data forward trough the network and returns the output
fn feedforward(a0: &Vector1x2, network: &Network) -> Vector1x1 {
    let z1: Vector1x2 = a0 * network.w1 + network.b1;
    let a1: Vector1x2 = z1.map(sigmoid);
    let z2: Vector1x1 = a1 * network.w2 + network.b2;
    let a2: Vector1x1 = z2.map(sigmoid);
    a2
}

//backpropagation algorithm 
fn backprop(a0: &Vector1x2, y: &Vector1x1, network: &Network) -> (Matrix2x2, Vector1x2, Vector2x1, Vector1x1) {
    //forward propagation
    let z1: Vector1x2 = a0 * network.w1 + network.b1;
    let a1: Vector1x2 = z1.map(sigmoid);
    let z2: Vector1x1 = a1 * network.w2 + network.b2;
    let a2: Vector1x1 = z2.map(sigmoid);
    
    //backward propagation
    let d2: Vector1x1 = -2.0 * (y - a2) * z2.map(sigmoid_prime);
    let dcdw2: Vector2x1 = a1.transpose() * d2;
    let dcdb2: Vector1x1 = d2;
    let d1: Vector1x2 = d2 * (network.w2.transpose().component_mul(&z1.map(sigmoid_prime)));
    let dcdw1: Matrix2x2 = a0.transpose() * d1;
    let dcdb1: Vector1x2 = d1;

    //gradient vectors
    (dcdw1, dcdb1, dcdw2, dcdb2)
}

//trains the network using gradient descent 
fn train(data: &Matrix4x3, network: &mut Network, learn_rate_start: f64, epoch_lenght: u32) {
    let mut learn_rate = learn_rate_start;
    let mut generation = 0;
    let mut epoch_start = 0;

    loop {
        generation += 1;
        if generation - epoch_start > epoch_lenght {
            epoch_start = generation;
            learn_rate /= 10.0;
        }

        //initialize gradient vectors 
        let mut dcdw1: Matrix2x2 = Matrix2x2::zeros();
        let mut dcdb1: Vector1x2 = Vector1x2::zeros();
        let mut dcdw2: Vector2x1 = Vector2x1::zeros();
        let mut dcdb2: Vector1x1 = Vector1x1::zeros();

        let mut avarage_cost = 0.0;

        //loop over training examples 
        for i in 0..data.nrows() {
            let a0: Vector1x2 = data.fixed_slice::<1, 2>(i, 0).into();
            let y: Vector1x1 = data.fixed_slice::<1, 1>(i, 2).into();
            
            let gradient: (Matrix2x2, Vector1x2, Vector2x1, Vector1x1) = backprop(&a0, &y, network);

            //add derivatives of current training example to the avarage gradient
            dcdw1 += gradient.0;
            dcdb1 += gradient.1;
            dcdw2 += gradient.2;
            dcdb2 += gradient.3;

            //compute cost for current training example 
            let a2 = feedforward(&a0, &network);
            let example_cost = cost(&a2, &y);
            avarage_cost += example_cost;
        }

        //avarage cost over all training examples
        avarage_cost /= data.nrows() as f64;
        println!("generation: {}, cost was {}, learn rate: {}", generation, avarage_cost, learn_rate);
        
        //update weights and biases using avarage gradient vector
        let num_of_examples = data.nrows() as f64;
        network.w1 -= learn_rate * dcdw1 / num_of_examples;
        network.b1 -= learn_rate * dcdb1 / num_of_examples;
        network.w2 -= learn_rate * dcdw2 / num_of_examples;
        network.b2 -= learn_rate * dcdb2 / num_of_examples;
    }
}

//mean squared error function
fn cost(a2: &Vector1x1, y: &Vector1x1) -> f64 {
    ((y - a2) * (y - a2))[(0, 0)]
}

//sigmoid activation function
fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + E.powf(-x))
}

//derivative of sigmoid activation function
fn sigmoid_prime(x: f64) -> f64 {
    sigmoid(x) * (1.0 - sigmoid(x))
}

fn main() {
    //XOR truth table
    let training_data: Matrix4x3 = Matrix4x3::new(0.0, 0.0, 0.0,
                                                  1.0, 0.0, 1.0,
                                                  0.0, 1.0, 1.0,
                                                  1.0, 1.0, 0.0);

    let mut ffnn: Network = init();

    train(&training_data, &mut ffnn, 1.0, 200_000);
}
