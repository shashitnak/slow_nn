#![allow(unused)]
#![deny(missing_docs)]

//! # Slow NN
//! 
//! A simple neural network library which uses a graph to build the network
//! as opposed to weight matrices, hence, is slow.
extern crate rand;
mod network;

pub use network::*;

#[cfg(test)]
#[allow(unused)]
mod tests {
    use super::network::*;
    use rand::Rng;
    
    fn random() -> f64 {
        rand::thread_rng().gen()
    }

    fn identity(value: f64) -> f64 {
        value
    }

    fn test_tuple(
        bias: f64,
        inputs: usize,
        outputs: usize,
        input: &[f64],
        tups: Vec<(usize, usize, f64)>,
    ) -> Vec<f64> {
        let conns: Vec<Connection> = tups.iter().map(|&t| t.into()).collect();
        let hidden = 3;
        let mut network = Network::from_conns(bias, inputs, outputs, hidden, &conns[..]);
        network.predict(input, identity)
        // let output = network.forward(&input, identity);
        // println!("{:?}", output);
    }

    #[test]
    #[test]
    fn test_network1() {
        let bias = 6.33;
        let mut input = vec![1.555, 0.3829, 120.023];
        input.reverse();
        let inputs = input.len();
        let outputs = 2;
        let conns = vec![
            (3, 6, 0.36),
            (2, 6, 0.20),
            (1, 6, 0.60),
            (0, 6, 0.12),
            (3, 7, 0.88),
            (2, 7, 0.52),
            (1, 7, 0.70),
            (0, 7, 0.14),
            (3, 8, 0.40),
            (2, 8, 0.32),
            (1, 8, 0.80),
            (0, 8, 0.16),
            (6, 4, 0.46),
            (7, 4, 0.70),
            (8, 4, 0.25),
            (6, 5, 0.55),
            (7, 5, 0.85),
            (8, 5, 0.50),
        ];
        let output = test_tuple(bias, inputs, outputs, &input, conns);
        assert_eq!(output, vec![118.7412964, 162.7625798]);

        let conns = vec![
            (3, 6, 0.36),
            (2, 6, 0.00),
            (1, 6, 0.60),
            (0, 6, 0.02),
            (3, 7, 0.88),
            (2, 7, 0.52),
            (1, 7, 0.70),
            (0, 7, 0.00),
            (3, 8, 0.00),
            (2, 8, 0.32),
            (1, 8, 0.80),
            (0, 8, 0.16),
            (6, 4, 0.40),
            (7, 4, 0.70),
            (8, 4, 0.05),
            (6, 5, 0.55),
            (7, 5, 0.00),
            (8, 5, 0.50),
        ];
        let output = test_tuple(bias, inputs, outputs, &input, conns);
        assert_eq!(output, vec![93.846292, 88.56197399999999]);
    }

    #[test]
    fn test_network2() {
        let bias = 0.5;
        let input = vec![0.3, 1.5];
        let inputs = input.len();
        let outputs = 1;
        let conns = vec![
            (0, 4, 1.0),
            (1, 4, 1.0),
            (2, 6, 1.0),
            (4, 3, 1.0),
            (4, 5, 1.0),
            (5, 6, 1.0),
            (6, 3, 1.0),
            (5, 4, 1.0),
            (6, 5, 1.0),
        ];

        test_tuple(bias, inputs, outputs, &input, conns);
    }

    #[test]
    fn test_dense() {
        let bias = 30. * random();
        let layers = [4];
        let mut net = Network::dense(bias, 2, 1, &layers);

        let _inputs = vec![vec![0., 0.], vec![0., 1.], vec![1., 0.], vec![1., 1.]];
        let outputs = vec![vec![0.], vec![1.], vec![1.], vec![1.]];

        let activation = sigmoid;
        let dactivation = dsigmoid;

        let lr = 0.1;

        for i in 0..=10000 {
            let mut preds = Vec::new();
            for i in 0..4 {
                let input = &_inputs[i];
                let output = &outputs[i];

                let pred = net.train(
                    input,
                    output,
                    activation,
                    dactivation,
                    loss,
                    dloss,
                    lr
                );

                preds.push(pred);
            }
            if random() < 0.1 || i == 1000 {
                println!("{:?}", preds);
            }
        }
    }

    #[test]
    fn test_xor() {
        let bias = 5.;
        let inputs = vec![vec![0., 0.], vec![0., 1.], vec![1., 0.], vec![1., 1.]];
        let outputs = vec![vec![1.], vec![1.], vec![1.], vec![0.]];

        //     [0.43040194, 0.21987024, 0.25842456],
        //    [0.02097203, 0.25657626, 0.73340706]

        //    [0.59851989, 0.3893478]

        let conns = vec![
            (0, 4, 0.43040194),
            (1, 4, 0.21987024),
            (2, 4, 0.25842456),
            (0, 5, 0.02097203),
            (1, 5, 0.25657626),
            (2, 5, 0.73340706),
            (0, 6, 0.23899292),
            (1, 6, 0.69989289),
            (2, 6, 0.89289283),
            (4, 3, 0.59851989),
            (5, 3, 0.3893478),
            (6, 3, 0.2293892),
        ];

        let conns: Vec<Connection> = conns.iter().map(|&t| t.into()).collect();

        let mut net = Network::from_conns(bias, 2, 1, 3, &conns);

        let lr = 0.10;

        for _ in 0..10000 {
            for i in 0..4 {
                let input = &inputs[i];
                let output = &outputs[i];
                let loss = net.train(input, output, sigmoid, dsigmoid, loss, dloss, lr);
            }
            // println!();
        }
        for i in 0..4 {
            let input = &inputs[i];
            let output = &outputs[i];
            let pred = net.predict(input, sigmoid);
            print!("{:?} ", pred);
        }
        println!();
    }

    fn loss(output: f64, expected: f64) -> f64 {
        (output - expected).powf(2.)
    }

    fn dloss(output: f64, expected: f64) -> f64 {
        2. * (output - expected)
    }

    fn sigmoid(x: f64) -> f64 {
        1. / (1. + (-x).exp())
    }

    fn dsigmoid(x: f64) -> f64 {
        x * (1. - x)
    }

    use std::fs::File;
    use std::io::Read;

    fn byte2int(bs: &[u8]) -> usize {
        ((bs[0] as usize) << 24)
            + ((bs[1] as usize) << 16)
            + ((bs[2] as usize) << 8)
            + (bs[3] as usize)
    }

    fn clean_image(buffer: Vec<u8>) -> Vec<f64> {
        buffer.iter().map(|&val| (val as f64) / 256.).collect()
    }

    fn clean_label(buffer: Vec<u8>) -> Vec<[f64; 10]> {
        buffer
            .iter()
            .map(|&val| {
                let mut arr = [0f64; 10];
                arr[val as usize] = 1.;
                arr
            })
            .collect()
    }

    fn read_train_images() -> Vec<u8> {
        let mut buffer = Vec::new();
        let path = "data/train-images-idx3-ubyte";
        let mut f = File::open(path).unwrap();
        let mut shit = [0u8; 16];
        f.read(&mut shit);
        // println!("{}", byte2int(&shit[0..4]));
        // println!("{}", byte2int(&shit[4..8]));
        // println!("{}", byte2int(&shit[8..12]));
        // println!("{}", byte2int(&shit[12..16]));
        f.read_to_end(&mut buffer);
        // println!("{}", buffer.len() / 784);
        buffer
    }

    fn read_test_images() -> Vec<u8> {
        let mut buffer = Vec::new();
        let path = "data/t10k-images-idx3-ubyte";
        let mut f = File::open(path).unwrap();
        let mut shit = [0u8; 16];
        f.read(&mut shit);
        // println!("{}", byte2int(&shit[0..4]));
        // println!("{}", byte2int(&shit[4..8]));
        // println!("{}", byte2int(&shit[8..12]));
        // println!("{}", byte2int(&shit[12..16]));
        f.read_to_end(&mut buffer);
        // println!("{}", buffer.len() / 784);
        buffer
    }

    fn read_train_labels() -> Vec<u8> {
        let mut buffer = Vec::new();
        let path = "data/train-labels-idx1-ubyte";
        let mut f = File::open(path).unwrap();
        let mut shit = [0u8; 8];
        f.read(&mut shit);
        // println!("{}", byte2int(&shit[0..4]));
        // println!("{}", byte2int(&shit[4..8]));
        f.read_to_end(&mut buffer);
        // println!("{}", buffer.len());
        buffer
    }

    fn read_test_labels() -> Vec<u8> {
        let mut buffer = Vec::new();
        let path = "data/t10k-labels-idx1-ubyte";
        let mut f = File::open(path).unwrap();
        let mut shit = [0u8; 8];
        f.read(&mut shit);
        // println!("{}", byte2int(&shit[0..4]));
        // println!("{}", byte2int(&shit[4..8]));
        f.read_to_end(&mut buffer);
        // println!("{}", buffer.len());
        buffer
    }

    fn print_image(img: &[u8]) {
        for i in 0..28 {
            for j in 0..28 {
                print!("{} ", (img[28 * i + j] != 0) as u8);
            }
            println!();
        }
    }

    use std::thread;
    use std::time::Duration;

    #[test]
    fn test_mnist() {
        let train_images = clean_image(read_train_images());
        // print_image(&train_images[0..784]);
        let test_images = clean_image(read_test_images());
        // print_image(&test_images[0..784]);
        let train_labels = clean_label(read_train_labels());
        let test_labels = clean_label(read_test_labels());

        let bias = 0.5 * random();
        let lr = 0.0001;
        let layers = [20, 20];
        let activation = sigmoid;
        let mut net = Network::dense(bias, 784, 10, &layers);

        for i in 0..60000 {
            let start = 784 * i;
            let end = 784 * (i + 1);
            let losss = net.train(
                &train_images[start..end],
                &train_labels[i],
                activation,
                dsigmoid,
                loss,
                dloss,
                lr
            );
            if (i + 1) % 6000 == 0 {
                println!("Loss = {}", losss);
            }
            // thread::sleep(Duration::from_nanos(10));
        }
        let mut count = 0.;
        for i in 0..10000 {
            let start = 784 * i;
            let end = 784 * (i + 1);
            let pred = net.predict(&test_images[start..end], activation);
            let mut max1 = 0.;
            let mut index1 = 0;
            let mut max2 = 0.;
            let mut index2 = 0;
            for j in 0..10 {
                if pred[j] > max1 {
                    max1 = pred[j];
                    index1 = j;
                }
                if test_labels[i][j] > max2 {
                    max2 = test_labels[i][j];
                    index2 = j;
                }
            }
            if index1 == index2 {
                count += 1.;
            }
        }
        println!("Accuracy = {}%", count / 100.);
    }
}
