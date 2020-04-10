mod neuron;

pub use neuron::*;
use std::collections::VecDeque;
use std::f64::NEG_INFINITY;
use serde::{Serialize, Deserialize};
use std::fs::File;
use std::io::{Read, Write, Error};

struct Outputs {
    outputs: Vec<f64>,
    sums: Vec<f64>,
}

impl Outputs {
    fn new(outputs: Vec<f64>, sums: Vec<f64>) -> Self {
        Self { outputs, sums }
    }
}

struct Gradients {
    grads: Vec<f64>,
    del_ws: Vec<Vec<f64>>,
}

impl Gradients {
    fn new(grads: Vec<f64>, del_ws: Vec<Vec<f64>>) -> Self {
        Self { grads, del_ws }
    }
}

/// Neural Network struct
#[derive(Serialize, Deserialize)]
pub struct Network {
    bias: f64,
    inputs: usize,
    outputs: usize,
    hidden: usize,
    neurons: Vec<Neuron>,
}

impl Network {
    /// Creates a network from a slice of Connections
    pub fn from_conns(
        bias: f64,
        inputs: usize,
        outputs: usize,
        hidden: usize,
        conns: &[Connection],
    ) -> Self {
        let cap = 1 + inputs + outputs + hidden;
        let mut neurons: Vec<_> = (0..).take(cap).map(|_| Neuron::new()).collect();

        for conn in conns {
            neurons[conn.to].connected_from(conn.from, conn.weight);
        }

        Self {
            bias,
            inputs,
            outputs,
            hidden,
            neurons,
        }
    }

    fn add_conns(
        conns: &mut Vec<Connection>,
        start1: usize,
        end1: usize,
        start2: usize,
        end2: usize,
    ) {
        for i in start1..end1 {
            for j in start2..end2 {
                conns.push(Connection::new(i, j));
            }
        }
    }

    fn rng2vec(start: usize, end: usize) -> Vec<usize> {
        (start..end).collect()
    }

    /// Creates a fully connected network with random weights
    pub fn dense(bias: f64, inputs: usize, outputs: usize, layers: &[usize]) -> Self {
        // Generate indeces for connections
        let mut offset = 1 + inputs + outputs;
        let mut indeces = vec![Self::rng2vec(0, 1 + inputs)];
        for layer in layers {
            indeces.push(Self::rng2vec(offset, offset + layer));
            offset += layer;
        }
        let mut conns = Vec::new();
        indeces.push(Self::rng2vec(1 + inputs, 1 + inputs + outputs));

        // Create connections
        for i in 1..indeces.len() {
            for &index1 in &indeces[i - 1] {
                for &index2 in &indeces[i] {
                    conns.push(Connection::new(index1, index2));
                }
            }
        }

        let hidden = offset - 1 - inputs - outputs;
        println!("Hidden = {}", hidden);

        Network::from_conns(bias, inputs, outputs, hidden, &conns)
    }

    /// Does a forward pass
    pub fn predict(&self, input: &[f64], activation: fn(f64) -> f64) -> Vec<f64> {
        let Outputs { outputs, .. } = self.compute_outputs(input, activation);
        let offset = 1 + self.inputs;

        outputs[offset..offset + self.outputs]
            .iter()
            .map(|&i| i)
            .collect()
    }

    /// Does both forward and backward pass and updates the weights
    pub fn train(
        &mut self,
        input: &[f64],
        expected: &[f64],
        activation: fn(f64) -> f64,
        deactivation: fn(f64) -> f64,
        loss: fn(f64, f64) -> f64,
        dloss: fn(f64, f64) -> f64,
        lr: f64,
    ) -> f64 {
        let outs = self.compute_outputs(input, activation);
        let pred: Vec<_> = outs.outputs.iter().skip(1 + self.inputs).take(self.outputs).cloned().collect();
        let error = pred.iter().zip(expected.iter()).fold(0., |acc, (&a, &b)| acc + loss(a, b));
        let Gradients { grads, del_ws } = self.compute_grads(outs, expected, deactivation, dloss, lr);

        self.bias += grads[0] * lr;
        for i in 0..self.neurons.len() {
            for j in 0..self.neurons[i].connections() {
                self.neurons[i].weights[j] -= del_ws[i][j] * lr;
            }
        }

        error
    }

    fn compute_outputs(&self, input: &[f64], activation: fn(f64) -> f64) -> Outputs {
        let len = self.neurons.len();
        let mut outputs: Vec<_> = (0..)
            .take(len)
            .map(|i| match i {
                0 => self.bias,
                x if x <= self.inputs => input[i - 1],
                _ => NEG_INFINITY,
            })
            .collect();
        let mut sums: Vec<_> = (0..)
            .take(len)
            .map(|_| 0.)
            .collect();
        let mut stack: Vec<_> = (0..).skip(1 + self.inputs).take(self.outputs).collect();

        while let Some(&top) = stack.last() {
            let len = stack.len();
            let mut sum = 0.0;
            for i in 0..self.neurons[top].connections() {
                let index = self.neurons[top].in_comes[i];
                if outputs[index] == NEG_INFINITY {
                    stack.push(index);
                } else {
                    let weight = self.neurons[top].weights[i];
                    sum += weight * outputs[index];
                }
            }

            if len == stack.len() {
                stack.pop();
                sums[top] = sum;
                outputs[top] = activation(sum);
            }
        }

        Outputs::new(outputs, sums)
    }

    fn compute_grads(
        &self,
        outs: Outputs,
        expected: &[f64],
        deactivation: fn(f64) -> f64,
        dloss: fn(f64, f64) -> f64,
        lr: f64,
    ) -> Gradients {
        let Outputs { outputs, sums } = outs;
        let offset = 1 + self.inputs;
        let mut grad: Vec<_> = (0..)
            .take(offset)
            .map(|_| NEG_INFINITY)
            .chain(
                (0..)
                    .take(self.outputs)
                    .map(|i| dloss(outputs[i + offset], expected[i])),
            )
            .chain((0..).take(self.hidden).map(|_| NEG_INFINITY))
            .collect();

        let mut del_ws: Vec<Vec<_>> = self
            .neurons
            .iter()
            .map(|n| (0..n.connections()).map(|_| 0.).collect())
            .collect();

        let mut queue: VecDeque<_> = (0..).skip(1 + self.inputs).take(self.outputs).collect();

        while let Some(top) = queue.pop_front() {
            let da = grad[top];
            let dz = da * deactivation(outputs[top]);

            for i in 0..self.neurons[top].connections() {
                let index = self.neurons[top].in_comes[i];

                let dw = dz * outputs[index];
                let dx = dz * self.neurons[top].weights[i];

                del_ws[top][i] += dw;

                if grad[index] == NEG_INFINITY {
                    grad[index] = dx;
                    queue.push_back(index);
                } else {
                    grad[index] += dx;
                }
            }
        }

        Gradients::new(grad, del_ws)
    }

    /// loads the network from a file
    pub fn load(path: &str) -> Result<Self, Error> {
        let mut file = File::open(path)?;
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer)?;
        Ok(bincode::deserialize(&buffer).expect("Error reading file"))
    }

    /// Saves the network to a file
    pub fn save(&self, path: &str) -> Result<(), Error> {
        let mut file = File::create(path)?;
        let buffer: Vec<u8> = bincode::serialize(&self).expect("Error saving file");
        file.write(&buffer)?;
        Ok(())
    }
}
