#![allow(unused)]

extern crate rand;

use rand::Rng;

pub fn random() -> f64 {
    rand::thread_rng().gen()
}

use std::f64::NEG_INFINITY;

pub struct Neuron {
    pub in_comes: Vec<usize>,
    pub weights: Vec<f64>,
}

impl Neuron {
    pub fn new() -> Self {
        Self {
            in_comes: Vec::new(),
            weights: Vec::new(),
        }
    }

    pub fn connected_from(&mut self, from: usize, weight: f64) {
        self.in_comes.push(from);
        self.weights.push(weight);
    }

    pub fn connections(&self) -> usize {
        self.in_comes.len()
    }
}

pub struct Connection {
    pub from: usize,
    pub to: usize,
    pub weight: f64,
}

impl Connection {
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
