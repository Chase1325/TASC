use time::Instant;
use rand::prelude::*;

use ndarray::s;
use ndarray::{Array, Axis, Ix1, Ix2};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;
use ndarray_stats::QuantileExt;
use rand::seq::SliceRandom;


pub trait DifferentialEvolution {

    fn get_bounds(&self) -> &Vec<(f64, f64)>;

    fn get_cost(&self, v: &[f64]) -> f64;

    fn update_opt(&mut self, params: &[f64], cost: &f64);

    fn minimum(&mut self) -> (Vec<f64>, f64) {

        //Diff evolution parameters
        let f: f64 = 0.8;
        let cr: f64 = 0.9;
        let max_iter: usize = 1000;

        //Create the population
        let bounds = self.get_bounds();
        let pop_size: usize = 25*bounds.len();
        let lb: Array<f64, Ix1> = Array::from_vec(bounds.iter().map(|&x| x.0).collect());
        let diffb: Array<f64, Ix1> = Array::from_vec(bounds.iter().map(|&x| x.1-x.0).collect());
        let mut pop: Array<f64, Ix2> = lb + Array::random((pop_size, bounds.len()), Uniform::new(0., 1.)) * diffb;

        //Initialize population objective solutions and bests
        let mut obj_all = Array::from_vec(pop.axis_iter(Axis(0)).map(|row| self.get_cost(&row.to_vec())).collect());
        let mut best_vec = pop.row(obj_all.argmin().unwrap()).to_vec();
        let mut best_obj = obj_all.min().unwrap().clone();
        let mut prev_obj = best_obj.clone();

        //Initialize candidates
        let candidates: Vec<usize> = (0..pop_size).collect();
        let mut candid_pop: [usize; 3] = [0; 3];

        //Begin optimization loop
        for i in 0..max_iter {
            for j in 0..pop_size {
    
                //choose 3 candidates, no replacement, not current j value
                let mut filled: usize = 0;
                while filled < 3 {
                    let rand_candid = *candidates.choose(&mut rand::thread_rng()).unwrap();
                    if rand_candid != j {
                        candid_pop[filled] = rand_candid;
                        filled += 1;
                    }
                }
                //Perform Mutation
                let a = pop.index_axis(Axis(0), candid_pop[0]);
                let b = pop.index_axis(Axis(0), candid_pop[1]);
                let c = pop.index_axis(Axis(0), candid_pop[2]);
                let mut mutation: Array<f64, Ix1> = &a + f * (&b-&c);
                for k in 0..bounds.len() {
                    mutation[k] = mutation[k].clamp(bounds[k].0, bounds[k].1);
                }
    
                //Perform Crossover
                let mut trial: Array<f64, Ix1> = mutation.clone(); 
                let mut target = pop.slice_mut(s![j, ..]);
                for k in 0..bounds.len() {
                    if rand::thread_rng().gen::<f64>() > cr {
                        trial[k] = target[k];
                    }
                }
    
                //Perform Selection
                let target_cost = self.get_cost(&target.to_vec());
                let trial_cost = self.get_cost(&trial.to_vec());
                if trial_cost < target_cost {
                    target.assign(&trial);
                    obj_all[j] = trial_cost;
                }
    
            //Find Best Agent
            best_obj = obj_all.min().unwrap().clone();
            if best_obj < prev_obj{
                if (obj_all.mean().unwrap().abs()-best_obj.abs()).abs() < 1e-3 && i > 25 {
                    best_vec = pop.row(obj_all.argmin().unwrap()).to_vec();
                    println!("{}", i);
                    self.update_opt(&best_vec, &best_obj);
                    return (best_vec, best_obj)
                }
                best_vec = pop.row(obj_all.argmin().unwrap()).to_vec();
                prev_obj = best_obj.clone();
            }
            }
        }
        (best_vec, best_obj)
    }
}