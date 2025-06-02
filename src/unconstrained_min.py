import numpy as np

class LineSearchOptimizer:
    def __init__(self, method_name="gradient_descent"):
        """
        Initializes the optimizer.
        Args:
            method_name (str): "gradient_descent" or "newton".
        """
        self.method_name = method_name.lower()
        if self.method_name not in ["gradient_descent", "newton"]:
            raise ValueError(
                "Unsupported method. Choose 'gradient_descent' or 'newton'."
            )
        self.path_history = []  # saves (x_i, f(x_i))
        self.last_iteration_details = None # saves details for reporting

    def _backtracking_line_search(self, f_obj, xk, pk, grad_fk, fk, 
                                  c1=0.01, beta=0.5, max_ls_iter=100):
        """
        Args:
            f_obj: The objective function (callable as f(x, hessian_needed=False)).
            xk: Current point.
            pk: Search direction.
            grad_fk: Gradient at xk.
            fk: Function value at xk.
            c1: Wolfe condition constant (assignment uses 0.01).
            beta: Reduction factor for alpha in each iteration (assignment uses 0.5).
            max_ls_iter: Maximum iterations for line search.

        Returns:
            alpha (float): The determined step size.
        """
        alpha = 1.0  
        
        # grad_fk.T @ pk is the directional derivative.
        # for a descent direction pk, this should be negative.
        directional_derivative = grad_fk.T @ pk

        for _ in range(max_ls_iter):
            f_new, _, _ = f_obj(xk + alpha * pk, hessian_needed=False)
            
            if f_new <= fk + c1 * alpha * directional_derivative:
                return alpha  # Wolfe condition satisfied
            
            alpha *= beta
        
        return alpha 

    def minimize(self, f_obj, x0, obj_tol, param_tol, max_iter, 
                 wolfe_c1=0.01, backtracking_beta=0.5):
        """
        minimizes f_obj using gradient descent or Newton's method
        Args:
            f_obj: the objective function to minimize, takes x and hessian_needed flag
            x0 : the starting point
            obj_tol : tolerance for objective function change or Newton decrement
            param_tol : tolerance for parameter change (norm of step)
            max_iter : maximum allowed number of iterations
            wolfe_c1 : c1 constant for Wolfe condition in line search
            backtracking_beta : beta constant for backtracking line search

        Returns:
            tuple: (x_final, f_final, success_flag)
                   x_final: final location.
                   f_final: final objective value.
                   success_flag: True if a termination criterion was met, False if max_iter reached or other failure.
        """
        self.path_history = [] # reset path history for new run
        x_k = np.array(x0, dtype=float) 

        f_k, grad_k, hess_k = f_obj(x_k, hessian_needed=(self.method_name == "newton"))
        
        self.path_history.append((x_k.copy(), f_k))
        print(f"Initial: x0 = {x_k}, f(x0) = {f_k:.6e}")
        self.last_iteration_details = ("Initial", 0, x_k, f_k, "initialization")

        for i in range(max_iter):
            iteration_num = i + 1
            x_prev = x_k.copy()
            f_prev = f_k

            if self.method_name == "gradient_descent":
                pk = -grad_k
            elif self.method_name == "newton":
                if hess_k is None: # shouldnt be met 
                    print("Warning: Hessian not available for Newton step, re-evaluating.")
                    _, _, hess_k = f_obj(x_k, hessian_needed=True)

                try:
                    pk_newton = np.linalg.solve(hess_k, -grad_k)
                    lambda_sq = -grad_k.T @ pk_newton 

                    if (lambda_sq / 2.0) < obj_tol:
                        report_msg = (
                            f"Termination: Newton decrement "
                            f"({lambda_sq / 2.0:.2e}) < obj_tol ({obj_tol:.2e})."
                        )
                        print(f"Iter {iteration_num:4d}: x = {x_k}, f(x) = {f_k:.6e}. {report_msg}")
                        self.last_iteration_details = ("Termination", iteration_num, x_k, f_k, report_msg)
                        return x_k, f_k, True
                    pk = pk_newton
                except np.linalg.LinAlgError:
                    print(f"Iter {iteration_num:4d}: Warning: Hessian is singular or ill-conditioned. "
                          "Switching to Gradient Descent step for this iteration.")
                    pk = -grad_k 
            else:
                raise ValueError("Internal error: LineSearchOptimizer/minimize")

            alpha = self._backtracking_line_search(
                f_obj, x_k, pk, grad_k, f_k, 
                c1=wolfe_c1, beta=backtracking_beta
            )
            
            x_k_plus_1 = x_k + alpha * pk

            f_k_plus_1, grad_k_plus_1, hess_k_plus_1 = f_obj(
                x_k_plus_1, hessian_needed=(self.method_name == "newton")
            )

            x_k = x_k_plus_1
            f_k = f_k_plus_1
            grad_k = grad_k_plus_1
            if self.method_name == "newton":
                hess_k = hess_k_plus_1 # store new Hessian for next iteration

            self.path_history.append((x_k.copy(), f_k))

            iter_report_msg = f"Iter {iteration_num:4d}: x = {x_k}, f(x) = {f_k:.6e}, alpha = {alpha:.4e}"
            print(iter_report_msg)
            self.last_iteration_details = ("Iteration", iteration_num, x_k, f_k, f"alpha = {alpha:.4e}")


            
            # objective change (2d)
            if abs(f_k - f_prev) < obj_tol:
                report_msg = (
                    f"Termination: Objective change "
                    f"({abs(f_k - f_prev):.2e}) < obj_tol ({obj_tol:.2e})."
                )
                print(report_msg)
                self.last_iteration_details = ("Termination", iteration_num, x_k, f_k, report_msg)
                return x_k, f_k, True 

            # parameter change (2e)
            if np.linalg.norm(x_k - x_prev) < param_tol:
                report_msg = (
                    f"Termination: Parameter change "
                    f"({np.linalg.norm(x_k - x_prev):.2e}) < param_tol ({param_tol:.2e})."
                )
                print(report_msg)
                self.last_iteration_details = ("Termination", iteration_num, x_k, f_k, report_msg)
                return x_k, f_k, True
        
        # looop finises once max_iter is reached
        report_msg = f"Termination: Maximum iterations ({max_iter}) reached."
        print(report_msg)
        self.last_iteration_details = ("Failure", max_iter, x_k, f_k, report_msg)
        return x_k, f_k, False 

    def get_path_history(self):
        return self.path_history

    def get_last_iteration_report_for_submission(self):
        """
        Returns a string for the final report as per submission instruction 6c.
        """
        if self.last_iteration_details:
            status, iter_num, x_final, f_final, reason_or_alpha = self.last_iteration_details
            
            x_final_str = np.array2string(x_final, precision=6, floatmode='fixed', suppress_small=True)

            if status == "Initial":
                 return f"Initial state: Iteration {iter_num}, x = {x_final_str}, f(x) = {f_final:.6e}"
            elif status == "Iteration": # usually means max_iter reached if this is the last
                 return f"Last console print (likely max_iter): Iteration {iter_num}, x = {x_final_str}, f(x) = {f_final:.6e}, {reason_or_alpha}"
            elif status == "Termination" or status == "Failure":
                 return f"Final state: Iteration {iter_num}, x = {x_final_str}, f(x) = {f_final:.6e}. Reason: {reason_or_alpha}"
        return "No optimization run completed or report data not set."