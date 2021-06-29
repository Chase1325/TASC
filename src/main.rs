extern crate differential_evolution;
use differential_evolution::self_adaptive_de;
use time::Instant;

fn main() {
    //fn rosen(v: &[f32]) -> f32 { (1. - v[0]).powf(2.) + 100.*(v[1]-v[0].powf(2.)).powf(2.)}
    fn eggholder(v: &[f32]) -> f32 {(-1. * (v[1] + 47.) * ((v[1] + v[0]/2. + 47.).abs().sqrt().sin())) - v[0] * ((v[0]-(v[1]+47.)).abs().sqrt().sin())}
    let initial_min_max = vec![(-512., 512.); 2];

    let mut de = self_adaptive_de(initial_min_max, eggholder);

    let start = Instant::now();
    de.iter().take(100000).find(|&cost| cost < -930.);
    let end = start.elapsed();
    let (cost, pos) = de.best().unwrap();
    println!("max value: {} found in [{}, {}], in {:.8?} seconds", cost, pos[0], pos[1], end);

}


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
