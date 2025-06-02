import unittest
import numpy as np
import os
import matplotlib.pyplot as plt
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.unconstrained_min import LineSearchOptimizer
from tests import examples 
from src import utils    

# Section 7
OBJ_TOL = 1e-12
PARAM_TOL = 1e-8
MAX_ITER_DEFAULT = 100
MAX_ITER_ROSENBROCK_GD = 10000 
X0_DEFAULT = np.array([1.0, 1.0])
X0_ROSENBROCK = np.array([-1.0, 2.0])
# Wolfe c1=0.01 and backtracking_beta=0.5 are defaults in LineSearchOptimizer

PLOT_DIR = "plots"
if not os.path.exists(PLOT_DIR):
    os.makedirs(PLOT_DIR)
    
class TestOptimizationRuns(unittest.TestCase):

    def _run_and_evaluate_optimizer(self, 
                                    optimizer_method_name, 
                                    obj_func, 
                                    x0, 
                                    max_iter,
                                    obj_func_name_str=""):
        """
        Helper function to run the optimizer and collect results.
        """
        print(f"\n--- Running {optimizer_method_name} on {obj_func_name_str} ---")
        print(f"Initial x0: {x0}")
        print(f"Max iterations: {max_iter}, Obj Tol: {OBJ_TOL}, Param Tol: {PARAM_TOL}")

        optimizer = LineSearchOptimizer(method_name=optimizer_method_name)
        
        # the wolfe_c1 and backtracking_beta are already set to 0.01 and 0.5as defaults in the LineSearchOptimizer's minimize method,
        # as required in 7.d 
        x_final, f_final, success = optimizer.minimize(
            f_obj=obj_func,
            x0=x0,
            obj_tol=OBJ_TOL,
            param_tol=PARAM_TOL,
            max_iter=max_iter
        )

        path_history = optimizer.get_path_history() # list of x_arr, f_val
        iterates_x = [item[0] for item in path_history] # list of x_k arrays
        f_values = [item[1] for item in path_history]   # list of f(x_k) values
        
        final_report_str = optimizer.get_last_iteration_report_for_submission()
        print(f"Optimizer Report for {optimizer_method_name} on {obj_func_name_str}:")
        print(final_report_str)
        print(f"Success flag: {success}")
        
        return iterates_x, f_values, final_report_str

    def _perform_plotting(self, obj_func, obj_func_name, 
                          plot_limits_x, plot_limits_y,
                          gd_path_x, gd_f_values, 
                          nt_path_x, nt_f_values):
        plot_filename_contour = os.path.join(PLOT_DIR, f"{obj_func_name.replace(' ', '_')}_contours.png")
        utils.plot_contour(
            obj_func=obj_func,
            x_limits=plot_limits_x,
            y_limits=plot_limits_y,
            paths=[
                (gd_path_x, "Gradient Descent", "blue"),
                (nt_path_x, "Newton", "red")
            ],
            title=f"Contour Plot: {obj_func_name}",
            show_plot=False
        )
        plt.savefig(plot_filename_contour)
        plt.close()
        print(f"Saved contour plot to {plot_filename_contour}")


        plot_filename_fvals = os.path.join(PLOT_DIR, f"{obj_func_name.replace(' ', '_')}_fvalues.png")
        utils.plot_func_values(
            iter_histories=[gd_f_values, nt_f_values],
            labels=["Gradient Descent", "Newton"],
            colors=["blue", "red"],
            title=f"Function Value vs. Iteration: {obj_func_name}",
            show_plot=False
        )
        plt.savefig(plot_filename_fvals)
        plt.close()
        print(f"Saved function value plot to {plot_filename_fvals}")
        print(f"--- Finished test for {obj_func_name} ---")


    def test_quadratic_circle(self):
        obj_func = examples.quadratic_example_circle
        obj_func_name = "Quadratic Circle"
        x0 = X0_DEFAULT
        plot_limits_x = (-1.5, 1.5)
        plot_limits_y = (-1.5, 1.5)
        
        gd_path_x, gd_f_values, _ = self._run_and_evaluate_optimizer(
            "gradient_descent", obj_func, x0, MAX_ITER_DEFAULT, obj_func_name)
        nt_path_x, nt_f_values, _ = self._run_and_evaluate_optimizer(
            "newton", obj_func, x0, MAX_ITER_DEFAULT, obj_func_name)
        
        self._perform_plotting(obj_func, obj_func_name, plot_limits_x, plot_limits_y,
                               gd_path_x, gd_f_values, nt_path_x, nt_f_values)

    def test_quadratic_axis_ellipses(self):
        obj_func = examples.quadratic_example_axis_ellipses
        obj_func_name = "Quadratic Axis-Aligned Ellipses"
        x0 = X0_DEFAULT
        plot_limits_x = (-1.5, 1.5) 
        plot_limits_y = (-0.5, 0.5) 
        
        gd_path_x, gd_f_values, _ = self._run_and_evaluate_optimizer(
            "gradient_descent", obj_func, x0, MAX_ITER_DEFAULT, obj_func_name)
        nt_path_x, nt_f_values, _ = self._run_and_evaluate_optimizer(
            "newton", obj_func, x0, MAX_ITER_DEFAULT, obj_func_name)

        self._perform_plotting(obj_func, obj_func_name, plot_limits_x, plot_limits_y,
                               gd_path_x, gd_f_values, nt_path_x, nt_f_values)

    def test_quadratic_rotated_ellipses(self):
        obj_func = examples.quadratic_example_rotated_ellipses
        obj_func_name = "Quadratic Rotated Ellipses"
        x0 = X0_DEFAULT
        plot_limits_x = (-1.5, 1.5) 
        plot_limits_y = (-1.5, 1.5)
        
        gd_path_x, gd_f_values, _ = self._run_and_evaluate_optimizer(
            "gradient_descent", obj_func, x0, MAX_ITER_DEFAULT, obj_func_name)
        nt_path_x, nt_f_values, _ = self._run_and_evaluate_optimizer(
            "newton", obj_func, x0, MAX_ITER_DEFAULT, obj_func_name)

        self._perform_plotting(obj_func, obj_func_name, plot_limits_x, plot_limits_y,
                               gd_path_x, gd_f_values, nt_path_x, nt_f_values)

    def test_rosenbrock(self):
        obj_func = examples.rosenbrock_function
        obj_func_name = "Rosenbrock"
        x0 = X0_ROSENBROCK 
        plot_limits_x = (-2, 2) 
        plot_limits_y = (-1, 3) 
        
        gd_path_x, gd_f_values, _ = self._run_and_evaluate_optimizer(
            "gradient_descent", obj_func, x0, MAX_ITER_ROSENBROCK_GD, obj_func_name) 
        nt_path_x, nt_f_values, _ = self._run_and_evaluate_optimizer(
            "newton", obj_func, x0, MAX_ITER_DEFAULT, obj_func_name)

        self._perform_plotting(obj_func, obj_func_name, plot_limits_x, plot_limits_y,
                               gd_path_x, gd_f_values, nt_path_x, nt_f_values)

    def test_linear_function(self):
        obj_func = examples.linear_function
        obj_func_name = "Linear Function"
        x0 = X0_DEFAULT
        plot_limits_x = (-5, 2) 
        plot_limits_y = (-10, 2) 
        
        gd_path_x, gd_f_values, _ = self._run_and_evaluate_optimizer(
            "gradient_descent", obj_func, x0, MAX_ITER_DEFAULT, obj_func_name)
        
        nt_path_x, nt_f_values, _ = self._run_and_evaluate_optimizer(
            "newton", obj_func, x0, MAX_ITER_DEFAULT, obj_func_name)

        self._perform_plotting(obj_func, obj_func_name, plot_limits_x, plot_limits_y,
                               gd_path_x, gd_f_values, nt_path_x, nt_f_values)

    def test_boyd_example_function(self):
        obj_func = examples.boyd_example_function
        obj_func_name = "Boyd Example Smoothed Triangle"
        x0 = X0_DEFAULT
        plot_limits_x = (-2, 2) 
        plot_limits_y = (-2, 2)
        
        gd_path_x, gd_f_values, _ = self._run_and_evaluate_optimizer(
            "gradient_descent", obj_func, x0, MAX_ITER_DEFAULT, obj_func_name)
        nt_path_x, nt_f_values, _ = self._run_and_evaluate_optimizer(
            "newton", obj_func, x0, MAX_ITER_DEFAULT, obj_func_name)

        self._perform_plotting(obj_func, obj_func_name, plot_limits_x, plot_limits_y,
                               gd_path_x, gd_f_values, nt_path_x, nt_f_values)


if __name__ == '__main__':
    unittest.main()
