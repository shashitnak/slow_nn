#![allow(unused)]

use rand::Rng;
use serde::{Serialize, Deserialize};

fn random() -> f64 {
    rand::thread_rng().gen()
}

use std::f64::NEG_INFINITY;

/// Neuron struct for Neural Network
#[derive(Serialize, Deserialize)]
pub struct Neuron {
    /// Indeces of incoming Neurons
    pub in_comes: Vec<usize>,
    /// Weights of Connections from respective incoming Neuron
    pub weights: Vec<f64>,
}

impl Neuron {
    /// Creates a new Neuron with no connections
    pub fn new() -> Self {
        Self {
            in_comes: Vec::new(),
            weights: Vec::new(),
        }
    }

    /// Adds a connection to the Neuron
    pub fn connected_from(&mut self, from: usize, weight: f64) {
        self.in_comes.push(from);
        self.weights.push(weight);
    }

    /// Returns the number of connections from the Neuron
    pub fn connections(&self) -> usize {
        self.in_comes.len()
    }
}

/// Connection struct for Neural Network
pub struct Connection {
    /// Index of Neuron from which the Connection is coming out of
    pub from: usize,
    /// Index of Neuron to which the Connection is going into
    pub to: usize,
    /// Weight of the Connection
    pub weight: f64,
}

impl Connection {
    /// Creates a new connection with random weight
    pub fn new(from: usize, to: usize) -> Self {
        Self {
            from,
            to,
            weight: 0.5 * random(),
        }
    }
}

impl Into<Connection> for (usize, usize, f64) {
    fn into(self) -> Connection {
        Connection {
            from: self.0,
            to: self.1,
            weight: self.2,
        }
    }
}
