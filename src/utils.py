import numpy as np
import matplotlib.pyplot as plt

def plot_contour(obj_func, 
                 x_limits=(-3, 3), 
                 y_limits=(-3, 3), 
                 num_points=100, 
                 levels=20,
                 title="Contour Plot",
                 paths=None, 
                 show_plot=True):
    """
    Plots the contour lines of a 2D objective function. Optionally overlays paths.

    Args:
        obj_func: the objective function to plot
        x_limits : min, max for x-axis
        y_limits : min, max for y-axis
        num_points : number of points in each dimension for the grid
        levels : number of contour levels or specific levels
        title : -
        paths : list of tuples, where each tuple contains:
            - path_data : list of 2D points (iterates)
            - label : label for the path in the legend
            - color : color for plotting the path
        show_plot : if True, calls plt.show()
    """
    x1_vals = np.linspace(x_limits[0], x_limits[1], num_points)
    x2_vals = np.linspace(y_limits[0], y_limits[1], num_points)
    X1, X2 = np.meshgrid(x1_vals, x2_vals)
    
    Z = np.zeros_like(X1)
    
    for i in range(X1.shape[0]):
        for j in range(X1.shape[1]):
            point = np.array([X1[i, j], X2[i, j]])
            f_val, _, _ = obj_func(point, hessian_needed=False)
            Z[i, j] = f_val

    plt.figure(figsize=(8, 6))
    contour = plt.contour(X1, X2, Z, levels=levels, cmap='viridis')
    plt.colorbar(contour, label='Function Value')
    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$")
    plt.title(title)
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.grid(True, linestyle=':', alpha=0.7)

    if paths:
        # Define styles for each path if you have a fixed number, Or pass them in as part of the 'paths' tuple
        line_styles = ['-', '--'] # Solid for first, dashed for second
        marker_styles = ['o', 'x'] # Circle for first, x for second
        marker_sizes = [3, 5]      # Different sizes can also help

        for i, (path_data, label, color) in enumerate(paths):
            if path_data:
                path_array = np.array(path_data)
                
                current_linestyle = line_styles[i % len(line_styles)]
                current_marker = marker_styles[i % len(marker_styles)]
                current_markersize = marker_sizes[i % len(marker_sizes)]

                plt.plot(path_array[:, 0], path_array[:, 1],
                         marker=current_marker,
                         markersize=current_markersize,
                         linestyle=current_linestyle,
                         label=label, color=color, linewidth=1.5)
        plt.legend()

    if show_plot:
        plt.show()

def plot_func_values(iter_histories, 
                     labels, 
                     colors,
                     title="Function Value vs. Iteration",
                     show_plot=True):
    """
    Plots function values at each iteration for one or more methods.

    Args:
        iter_histories (list of lists): Each inner list contains f(x_i) values
                                        for a method.
        labels (list of str): Labels for each method.
        colors (list of str): Colors for each method's plot.
        title (str): Plot title.
        show_plot (bool): If True, calls plt.show().
    """
    if not (len(iter_histories) == len(labels) == len(colors)):
        raise ValueError("iter_histories, labels, and colors must have the same length.")

    # Define styles for each path if you have a fixed number, Or pass them in as part of the 'paths' tuple
    line_styles = ['-', '--'] # Solid for first, dashed for second
    marker_styles = ['o', 'x'] # Circle for first, x for second
    marker_sizes = [3, 5]      # Different sizes can also help
    plt.figure(figsize=(10, 6))
    for i, f_values in enumerate(iter_histories):
        iterations = range(len(f_values))
        current_linestyle = line_styles[i % len(line_styles)]
        current_marker = marker_styles[i % len(marker_styles)]
        current_markersize = marker_sizes[i % len(marker_sizes)]
        plt.plot(iterations, f_values, label=labels[i], color=colors[i], marker=current_marker, markersize=current_markersize, linestyle=current_linestyle)
    
    plt.xlabel("Iteration Number")
    plt.ylabel("Objective Function Value $f(x_i)$")
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.yscale('log') 
    
    if show_plot:
        plt.show()