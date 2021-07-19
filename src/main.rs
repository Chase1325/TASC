extern crate differential_evolution;
use differential_evolution::self_adaptive_de;
use time::Instant;
use simplers_optimization::Optimizer;
use argmin::prelude::*;
use argmin::solver::particleswarm::*;
use argmin::solver::simulatedannealing::*;
use num::{Float, FromPrimitive};
use argmin_testfunctions::{himmelblau, eggholder};
use rand::prelude::*;
use rand_xorshift::XorShiftRng;
use std::sync::Arc;
use std::sync::Mutex;

use ndarray::s;
use ndarray::{Array, Axis, Ix1};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;
use ndarray_stats::QuantileExt;
use rand::seq::SliceRandom;

struct Testfunc {}

pub fn testfunc<T: Float + FromPrimitive>(param: &[T]) -> T {
    let n47 = T::from_f32(47.0).unwrap();
    let n2 = T::from_f32(2.0).unwrap();
    let pi = T::from_f32(3.14159).unwrap();
    //(-(param[1] + n47) * ((param[1] + param[0]/n2 + n47).abs().sqrt().sin())) - param[0] * ((param[0]-(param[1]+n47)).abs().sqrt().sin())
    -(param[0]).cos()*(param[1]).cos()*(-((param[0]-pi).powi(2) + (param[1]-pi).powi(2))).exp()
}

impl ArgminOp for Testfunc {
    type Param = Vec<f32>;
    type Output = f32;
    type Hessian = ();
    type Jacobian = ();
    type Float = f32;

    fn apply(&self, param: &Self::Param) -> Result<Self::Output, Error> {
        Ok(testfunc(param))
    }
}  

struct Himmelblau {}//{upper_bound: Vec<f64>, lower_bound: Vec<f64>, rng: Arc<Mutex<XorShiftRng>>}

impl ArgminOp for Himmelblau {
    type Param = Vec<f64>;
    type Output = f64;
    type Hessian = ();
    type Jacobian = ();
    type Float = f64;

    fn apply(&self, param: &Self::Param) -> Result<Self::Output, Error> {
        Ok(eggholder(param))
    }
    /*
    fn modify(&self, param: &Vec<f64>, temp: f64) -> Result<Vec<f64>, Error> {
        let mut param_n = param.clone();
        // Perform modifications to a degree proportional to the current temperature `temp`.
        for _ in 0..(temp.floor() as u64 + 1) {
            // Compute random index of the parameter vector using the supplied random number
            // generator.
            let mut rng = self.rng.lock().unwrap();//let mut rng = rand::thread_rng();//self.rng.lock().unwrap();
            let idx = (*rng).gen_range(0..param.len());

            // Compute random number in [0.1, 0.1].
            let val = 0.1 * (*rng).gen_range(-1.0..1.0);

            // modify previous parameter value at random position `idx` by `val`
            let tmp = param[idx] + val;

            // check if bounds are violated. If yes, project onto bound.
            if tmp > self.upper_bound[idx] {
                param_n[idx] = self.upper_bound[idx];
            } else if tmp < self.lower_bound[idx] {
                param_n[idx] = self.lower_bound[idx];
            } else {
                param_n[idx] = param[idx] + val;
            }
        }
        Ok(param_n)
    }*/
}

fn run() -> Result<(), Error> {
    // Define inital parameter vector
    let init_param: Vec<f32> = vec![510.1, 0.1];

    let cost_function = Testfunc {};//{lower_bound: vec![-512., -512.], upper_bound: vec![512., 512.], rng: Arc::new(Mutex::new(XorShiftRng::from_entropy()))};

    //let solver = SimulatedAnnealing::new(15.)?.reannealing_fixed(1000);//.stall_best(10000).stall_accepted(100_000).reannealing_accepted(100);
    let solver = ParticleSwarm::new((vec![-100., -100.], vec![100.0, 100.0]), 20, 0.5, 0.0, 0.5)?;

    let executor = Executor::new(cost_function, solver, init_param).max_iters(10000);

    let res = executor.run()?;

    println!("{}", res);
    

    Ok(())
}


fn evolution<F>(bounds: Vec<(f64, f64)>, obj_func: F) -> (Vec<f64>, f64) where F: Fn(&[f64]) -> f64 {
    let pop_size: usize = 25*bounds.len();
    let f: f64 = 0.8;
    let cr: f64 = 0.9;
    let max_iter: usize = 1000;

    let lb = Array::from_vec(bounds.iter().map(|&x| x.0).collect());
    let diffb = Array::from_vec(bounds.iter().map(|&x| x.1-x.0).collect());


    let mut pop = Array::random((pop_size, bounds.len()), Uniform::new(0., 1.));
    pop = lb + pop * diffb;

    let mut obj_all = Array::from_vec(pop.axis_iter(Axis(0)).map(|row| obj_func(&row.to_vec())).collect());

    let mut best_vec = pop.row(obj_all.argmin().unwrap()).to_vec();
    let mut best_obj = obj_all.min().unwrap().clone();
    let mut prev_obj = best_obj.clone();

    let candidates: Vec<usize> = (0..pop_size).collect();
    let mut candid_pop: [usize; 3] = [0; 3];

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
            let target_cost = obj_func(&target.to_vec());
            let trial_cost = obj_func(&trial.to_vec());
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
                return (best_vec, best_obj)
            }
            best_vec = pop.row(obj_all.argmin().unwrap()).to_vec();
            prev_obj = best_obj.clone();
        }
        }
    }
    (best_vec, best_obj)
}

fn main() {
    fn eggholder(v: &[f64]) -> f64 {(-1. * (v[1] + 47.) * ((v[1] + v[0]/2. + 47.).abs().sqrt().sin())) - v[0] * ((v[0]-(v[1]+47.)).abs().sqrt().sin())}
    fn himmel(v: &[f64]) -> f64 {(v[0].powi(2) + v[1]-11.).powi(2) + (v[0] + v[1].powi(2) -7.).powi(2)}
    fn easom(v: &[f64]) -> f64 {-(v[0]).cos()*(v[1].cos()*(-((v[0]-std::f64::consts::PI).powi(2) + (v[1]-std::f64::consts::PI).powi(2))).exp())}
    let initial_min_max = vec![(-100., 100.); 2];

  
    let initial_min_max = vec![(-512., 512.); 2];

    let start = Instant::now();
    let (max_val, coord) = Optimizer::minimize(&easom, &initial_min_max, 10000);//Optimizer::new(&eggholder, &initial_min_max, true).set_exploration_depth(5).skip(100).skip_while(|(value, coords)| *value > -958.).next().unwrap();
    let end = start.elapsed();
    println!("max value: {} found in [{}, {}], in {:.8?} seconds", max_val, coord[0], coord[1], end);
    //if let Err(ref e) = run() {
    //    println!("{}", e);
    //}
    let start = Instant::now();
    let res = evolution(initial_min_max, &himmel);
    let end = start.elapsed();
    println!("{:?}", &res);
    println!("{:?} seconds", end.as_seconds_f64());
}

/*pub struct Testfunc {}

pub fn testfunc<T: Float + FromPrimitive>(param: &[T]) -> T {
    let n47 = T::from_f32(47.0).unwrap();
    let n2 = T::from_f32(2.0).unwrap();
    (-(param[1] + n47) * ((param[1] + param[0]/n2 + n47).abs().sqrt().sin())) - param[0] * ((param[0]-(param[1]+n47)).abs().sqrt().sin())
}
struct Himmelblau {}

impl ArgminOp for Himmelblau {
    type Param = Vec<f32>;
    type Output = f32;
    type Hessian = ();
    type Jacobian = ();
    type Float = f32;

    fn apply(&self, param: &Self::Param) -> Result<Self::Output, Error> {
        Ok(himmelblau(param))
    }
}  

fn main() {
    //fn rosen(v: &[f32]) -> f32 { (1. - v[0]).powf(2.) + 100.*(v[1]-v[0].powf(2.)).powf(2.)}
    fn eggholder(v: &[f32]) -> f32 {(-1. * (v[1] + 47.) * ((v[1] + v[0]/2. + 47.).abs().sqrt().sin())) - v[0] * ((v[0]-(v[1]+47.)).abs().sqrt().sin())}
    let initial_min_max = vec![(-512.01, 512.01); 2];

    let mut de = self_adaptive_de(initial_min_max, eggholder);

    let start = Instant::now();
    de.iter().take(100000).find(|&cost| cost < -930.);
    let end = start.elapsed();
    let (cost, pos) = de.best().unwrap();
    println!("max value: {} found in [{}, {}], in {:.8?} seconds", cost, pos[0], pos[1], end);

    let initial_min_max = vec![(-512., 512.); 2];

    let start = Instant::now();
    let (max_val, coord) = Optimizer::minimize(&eggholder, &initial_min_max, 1000);//Optimizer::new(&eggholder, &initial_min_max, true).set_exploration_depth(5).skip(100).skip_while(|(value, coords)| *value > -958.).next().unwrap();
    let end = start.elapsed();
    println!("max value: {} found in [{}, {}], in {:.8?} seconds", max_val, coord[0], coord[1], end);

    //let solver = argmin::solver::simulatedannealing::SimulatedAnnealing::new(15.);
    //let res = Executor::new(eggholder, solver, )
    let cost_func = Himmelblau {};
    let position: (Vec<f32>, Vec<f32>) = (vec![-512., -512.], vec![512., 512.]);
    let solver = ParticleSwarm::new(position, 100, 0.5, 0.0, 0.5)?;
    let init_param: Vec<f32> = vec![0.1, 0.1];
    let rest = Executor::new(cost_func, solver, init_param);
}

*/
/*use argmin::prelude::*;
use argmin::solver::simulatedannealing::{SATempFunc, SimulatedAnnealing};
use argmin_testfunctions::rosenbrock;
use rand::prelude::*;
use rand_xorshift::XorShiftRng;
use std::default::Default;
use std::sync::Arc;
use std::sync::Mutex;

struct Rosenbrock {
    /// Parameter a, usually 1.0
    a: f64,
    /// Parameter b, usually 100.0
    b: f64,
    /// lower bound
    lower_bound: Vec<f64>,
    /// upper bound
    upper_bound: Vec<f64>,
    /// Random number generator. We use a `Arc<Mutex<_>>` here because `ArgminOperator` requires
    /// `self` to be passed as an immutable reference. This gives us thread safe interior
    /// mutability.
    rng: Arc<Mutex<XorShiftRng>>,
}

impl Default for Rosenbrock {
    fn default() -> Self {
        let lower_bound: Vec<f64> = vec![-5.0, -5.0];
        let upper_bound: Vec<f64> = vec![5.0, 5.0];
        Rosenbrock::new(1.0, 100.0, lower_bound, upper_bound)
    }
}

impl Rosenbrock {
    /// Constructor
    pub fn new(a: f64, b: f64, lower_bound: Vec<f64>, upper_bound: Vec<f64>) -> Self {
        Rosenbrock {
            a,
            b,
            lower_bound,
            upper_bound,
            rng: Arc::new(Mutex::new(XorShiftRng::from_entropy())),
        }
    }
}

impl ArgminOp for Rosenbrock {
    type Param = Vec<f64>;
    type Output = f64;
    type Hessian = ();
    type Jacobian = ();
    type Float = f64;

    fn apply(&self, param: &Vec<f64>) -> Result<f64, Error> {
        Ok(rosenbrock(param, self.a, self.b))
    }

    /// This function is called by the annealing function
    fn modify(&self, param: &Vec<f64>, temp: f64) -> Result<Vec<f64>, Error> {
        let mut param_n = param.clone();
        // Perform modifications to a degree proportional to the current temperature `temp`.
        for _ in 0..(temp.floor() as u64 + 1) {
            // Compute random index of the parameter vector using the supplied random number
            // generator.
            let mut rng = self.rng.lock().unwrap();
            let idx = (*rng).gen_range(0..param.len());

            // Compute random number in [0.1, 0.1].
            let val = 0.1 * (*rng).gen_range(-1.0..1.0);

            // modify previous parameter value at random position `idx` by `val`
            let tmp = param[idx] + val;

            // check if bounds are violated. If yes, project onto bound.
            if tmp > self.upper_bound[idx] {
                param_n[idx] = self.upper_bound[idx];
            } else if tmp < self.lower_bound[idx] {
                param_n[idx] = self.lower_bound[idx];
            } else {
                param_n[idx] = param[idx] + val;
            }
        }
        Ok(param_n)
    }
}

fn run() -> Result<(), Error> {
    // Define bounds
    let lower_bound: Vec<f64> = vec![-5.0, -5.0];
    let upper_bound: Vec<f64> = vec![5.0, 5.0];

    // Define cost function
    let operator = Rosenbrock::new(1.0, 100.0, lower_bound, upper_bound);

    // definie inital parameter vector
    let init_param: Vec<f64> = vec![3.0, 1.2];

    // Define initial temperature
    let temp = 15.0;

    // Set up simulated annealing solver
    let solver = SimulatedAnnealing::new(temp)?
        // Optional: Define temperature function (defaults to `SATempFunc::TemperatureFast`)
        .temp_func(SATempFunc::Boltzmann)
        /////////////////////////
        // Stopping criteria   //
        /////////////////////////
        // Optional: stop if there was no new best solution after 1000 iterations
        .stall_best(10000)
        // Optional: stop if there was no accepted solution after 1000 iterations
        .stall_accepted(10000)
        /////////////////////////
        // Reannealing         //
        /////////////////////////
        // Optional: Reanneal after 1000 iterations (resets temperature to initial temperature)
        .reannealing_fixed(10000)
        // Optional: Reanneal after no accepted solution has been found for `iter` iterations
        .reannealing_accepted(5000)
        // Optional: Start reannealing after no new best solution has been found for 800 iterations
        .reannealing_best(8000);

    /////////////////////////
    // Run solver          //
    /////////////////////////
    let res = Executor::new(operator, solver, init_param)
        // Optional: Attach a observer
        //.add_observer(ArgminSlogLogger::term(), ObserverMode::Always)
        // Optional: Set maximum number of iterations (defaults to `std::u64::MAX`)
        .max_iters(100_000)
        // Optional: Set target cost function value (defaults to `std::f64::NEG_INFINITY`)
        .target_cost(0.0)
        .run()?;

    // Wait a second (lets the logger flush everything before printing again)
    //std::thread::sleep(std::time::Duration::from_secs(1));

    // Print result
    println!("{}", res);
    Ok(())
}

fn main() {
    if let Err(ref e) = run() {
        //println!("{}", e);
        std::process::exit(1);
    }
}
*/
/*use simplers_optimization::Optimizer;
use time::Instant;

fn main() {
    let f = |v:&[f64]| (-1. * (v[1] + 47.) * (v[1] + v[0]/2. + 47.).abs().sqrt().sin()) - v[0] * (v[0]-(v[1]+47.)).abs().sqrt().sin();
    let rosen = |v:&[f64]| (1. - v[0]).powf(2.) + 100.*(v[1]-v[0].powf(2.)).powf(2.);
    let input_interval = vec![(-2., 2.), (-3., 3.)];
    let nb_iterations = 100000;

    let start = Instant::now();
    let (max_value, coordinates) = Optimizer::minimize(&rosen, &input_interval, nb_iterations);
    let end = start.elapsed();
    println!("max value: {} found in [{}, {}], in {:.8?} seconds", max_value, coordinates[0], coordinates[1], end);
}*/
