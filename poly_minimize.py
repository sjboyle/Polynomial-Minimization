import tkinter
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mpl_toolkits.mplot3d import axes3d
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import cm
from matplotlib.ticker import LinearLocator, MultipleLocator, FormatStrFormatter

matplotlib.use('TkAgg')


class Polynomial:  # Quadratic Polynomial in x and y with 6 coefficients

    def __init__(self, coefficients):
        self.coeff = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.coeff
        n_coeff = len(coefficients)
        if n_coeff > 6:
            n_coeff = 6
        self.coeff[:n_coeff] = coefficients[:n_coeff]

    def __str__(self):
        out = ""
        prev = False
        coeff_text = ["", "y", "x", "xy", "y^2", "x^2"]
        for i in range(len(self.coeff) - 1, -1, -1):
            if self.coeff[i] != 0.0:
                if prev:
                    out = out + " + "
                prev = True
                out = out + str(self.coeff[i]) + coeff_text[i]
        return out

    def evaluate(self, x, y):
        return (self.coeff[0] + self.coeff[1] * y + self.coeff[2] * x +
                self.coeff[3] * x * y + self.coeff[4] * y * y + self.coeff[5] * x * x)

    def gradient(self, x, y):
        grad_x = 2 * self.coeff[5] * x + self.coeff[3] * y + self.coeff[2]
        grad_y = 2 * self.coeff[4] * y + self.coeff[3] * x + self.coeff[1]
        return grad_x, grad_y

    def minimize(self, x, y, alpha):  # Use gradient descent to find minimum
        delta = 0.000001  # accuracy
        max_iter = 1000  # Maximum number of iterations
        i = 1
        point_sequence = [(x, y)]  # Start point of iteration
        x_old = x
        y_old = y
        while True:
            if i > max_iter:
                break
            grad = self.gradient(x_old, y_old)
            x_new = x_old - alpha * grad[0]  # New estimate for minimum
            y_new = y_old - alpha * grad[1]
            point_sequence.append((x_new, y_new))
            if (abs(x_old - x_new) < delta) and (abs(y_old - y_new) < delta):
                break
            i += 1
            x_old = x_new
            y_old = y_new
        p_min = (x_new, y_new)      # Final convergence point
        if i > max_iter:
            out = "No Convergence"
        else:                       # Classify type of minimum
            point_type = (4 * self.coeff[5] * self.coeff[4]) - (self.coeff[3] * self.coeff[3])
            if point_type > delta:
                out = "Local Minimum"
            elif point_type < -delta:
                out = "Saddle Point"
            else:
                out = "Unable to Classify Point"
        return p_min, point_sequence, out


class PolyProgram:

    def __init__(self):
        self.fig = None
        self.figure_canvas = None
        self.coeff_entries = []
        self.coeff_labels = []
        self.coeff_label_text = ["x^2 +", "y^2 +", "xy +", "x +", "y +"]
        self.window = tkinter.Tk()
        self.window.title("Polynomial Minimization")
        self.canvas = tkinter.Canvas(self.window, width=700, height=600, bg="#FFFFFF")
        self.frame_low_left = tkinter.Frame(self.window, bg="#FFFFFF")
        self.frame_low_mid = tkinter.Frame(self.window, bg="#F0F0F0")
        self.frame_poly = tkinter.Frame(self.window, bg="#FFFFFF")
        self.frame_startpt = tkinter.Frame(self.frame_low_mid, bg="#FFFFFF")
        self.poly_label = tkinter.Label(self.frame_poly, text="Polynomial: ", font=("Helvetica", 10),
                                        bg="#FFFFFF", fg="#000000")
        for i in range(6):
            self.coeff_entries.append(tkinter.Entry(self.frame_poly, width=5))
            self.coeff_entries[i].insert(0, "0.0")
            self.coeff_labels.append(tkinter.Label(self.frame_poly, text="",
                                                   font=("Helvetica", 10), bg="#FBFBFB", fg="#000000"))
            if i != 5:
                self.coeff_labels[i].config(text=self.coeff_label_text[i])
        self.poly_label.pack(pady=5, side=tkinter.LEFT)
        for i in range(6):
            self.coeff_entries[i].pack(side=tkinter.LEFT)
            if i != 5:
                self.coeff_labels[i].pack(side=tkinter.LEFT)
        self.plot_button = tkinter.Button(self.frame_low_left, text="Plot Function",
                                          padx=0, pady=0, command=self.plot_surf)
        self.min_button = tkinter.Button(self.frame_low_mid, text="Minimize Function",
                                         padx=0, pady=0, command=self.min)
        self.min_label_1 = tkinter.Label(self.frame_startpt, text="Start Point for Minimization X,Y: ",
                                         font=("Helvetica", 10), bg="#FBFBFB", fg="#000000")
        self.min_label_2 = tkinter.Label(self.frame_startpt, text=",",
                                         font=("Helvetica", 10), bg="#FBFBFB", fg="#000000")
        self.x_entry = tkinter.Entry(self.frame_startpt, width=5)
        self.y_entry = tkinter.Entry(self.frame_startpt, width=5)
        self.x_entry.insert(0, 0.0)
        self.y_entry.insert(0, 0.0)
        self.min_text = tkinter.Text(self.window, height=4, width=30)
        self.plot_button.pack(side=tkinter.LEFT, ipadx=0)
        self.min_label_1.pack(side=tkinter.LEFT)
        self.x_entry.pack(side=tkinter.LEFT)
        self.min_label_2.pack(side=tkinter.LEFT)
        self.y_entry.pack(side=tkinter.LEFT)
        self.min_button.pack(ipadx=0)
        self.frame_startpt.pack(pady=10)
        self.canvas.pack()
        self.frame_poly.pack(pady=5)
        self.frame_low_left.pack(side=tkinter.LEFT, padx=30, pady=10)
        self.frame_low_mid.pack(side=tkinter.LEFT, pady=10)
        self.min_text.pack(side=tkinter.LEFT, padx=10, pady=10)

    def run(self):
        self.window.mainloop()

    def get_polynomial_coeffs(self):  # Get polynomial coefficients from Entry fields
        coeffs = []
        for i in range(5, -1, -1):
            inp = self.coeff_entries[i].get()
            try:
                coeffs.append(float(inp))
            except ValueError:
                coeffs.append(0.0)
        return coeffs

    def get_start_point(self):  # Get star point from Entry fields
        try:
            x = float(self.x_entry.get())
        except ValueError:
            x = 0.0
        try:
            y = float(self.y_entry.get())
        except ValueError:
            x = 0.0
        return x, y

    def plot_surf(self):
        self.plot([])

    def plot(self, path):
        if self.figure_canvas is not None:  # delete any existing plot
            for item in self.figure_canvas.get_tk_widget().find_all():
                self.figure_canvas.get_tk_widget().delete(item)
            self.figure_canvas.get_tk_widget().pack()
            self.figure_canvas.get_tk_widget().destroy()
        print("plotting")
        poly = Polynomial(self.get_polynomial_coeffs())
        self.fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        self.figure_canvas = FigureCanvasTkAgg(self.fig, master=self.canvas)
        self.figure_canvas.get_tk_widget().pack()
        self.figure_canvas.draw()
        X = np.arange(-50, 55, 5)
        Y = np.arange(-50, 55, 5)
        X, Y = np.meshgrid(X, Y)            # Create 2D grid of X and Y coordinates
        Z = np.zeros_like(X)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                Z[i, j] = poly.evaluate(X[i, j], Y[i, j])

        # Plot minimization path
        path_points = np.array(path)
        if path_points.size != 0:
            xp = path_points[:, 0]      # Extract X coordinates
            yp = path_points[:, 1]      # Extract Y coordinates
            zp = np.zeros_like(xp)
            for i in range(zp.size):
                zp[i] = poly.evaluate(xp[i], yp[i])

        # Plot the surface
        surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                               linewidth=0, antialiased=False)
        if path_points.size != 0:
            ax.plot(xp, yp, zp, 'k+', alpha=0.5)

        # Customize the z axis.
        # ax.set_zlim(-100, 100)
        ax.zaxis.set_major_locator(LinearLocator(10))
        # A StrMethodFormatter is used automatically
        ax.zaxis.set_major_formatter('{x:.02f}')

        # Add a color bar which maps values to colors.
        self.fig.colorbar(surf, shrink=0.5, aspect=5)
        # plt.show()
        plt.close(self.fig)

    def min(self):
        self.plot([])
        self.min_text.delete(1.0, tkinter.END)
        poly = Polynomial(self.get_polynomial_coeffs())
        (s_x, s_y) = self.get_start_point()
        min_pt, points, result = poly.minimize(s_x, s_y, 0.01)
        txt = result
        if result != "No Convergence":
            txt = txt + "\nMinimum at ({:.3f}, {:.3f})".format(min_pt[0], min_pt[1])
            self.plot(points)
        self.min_text.insert(1.0, txt)


polyProgram = PolyProgram()
polyProgram.run()
