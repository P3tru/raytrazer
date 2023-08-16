import numpy as np
from math import isclose, sqrt

from dataclasses import dataclass
from typing import List, Tuple, Optional

import matplotlib.pyplot as plt


@dataclass
class Vector2:
    """
    A class representing a 2D vector.

    Attributes:
        x (float): The x-coordinate of the vector.
        y (float): The y-coordinate of the vector.
    """

    x: float
    y: float

    def __add__(self, other):
        """
        Adds this vector to another vector.

        Args:
            other (Vector2): The vector to be added.

        Returns:
            Vector2: The resulting vector after addition.

        Raises:
            TypeError: If the operand type for addition is not supported.
        """
        if isinstance(other, Vector2):
            return Vector2(self.x + other.x, self.y + other.y)
        raise TypeError("Unsupported operand type for +")

    def __sub__(self, other):
        """
        Subtracts another vector from this vector.

        Args:
            other (Vector2): The vector to be subtracted.

        Returns:
            Vector2: The resulting vector after subtraction.

        Raises:
            TypeError: If the operand type for subtraction is not supported.
        """
        if isinstance(other, Vector2):
            return Vector2(self.x - other.x, self.y - other.y)
        raise TypeError("Unsupported operand type for -")

    def __mul__(self, other):
        """
        Multiplies this vector by a scalar value or another vector.

        Args:
            other (Union[int, float, Vector2]): The scalar value or vector to be multiplied.

        Returns:
            Vector2: The resulting vector after multiplication.

        Raises:
            TypeError: If the operand type for multiplication is not supported.
        """
        if isinstance(other, (int, float)):
            return Vector2(self.x * other, self.y * other)
        raise TypeError("Unsupported operand type for *")

    def __rmul__(self, other):
        """
        Multiplies this vector by a scalar value from the right side.

        Args:
            other (Union[int, float]): The scalar value to be multiplied.

        Returns:
            Vector2: The resulting vector after multiplication.

        Raises:
            TypeError: If the operand type for multiplication is not supported.
        """
        return self.__mul__(other)

    def __getitem__(self, index):
        """
        Accesses the vector coordinates using square brackets.

        Args:
            index (int): The index of the coordinate (0 for x, 1 for y).

        Returns:
            float: The value of the coordinate.

        Raises:
            IndexError: If the index is out of range.
        """
        if index == 0:
            return self.x
        elif index == 1:
            return self.y
        raise IndexError("Vector index out of range")

    def __iter__(self):
        """
        Iterates over the vector coordinates.

        Yields:
            float: The value of each coordinate.
        """
        yield self.x
        yield self.y

    def normalize(self):
        """
        Computes the unit vector in the same direction as this vector.

        Returns:
            Vector2: The normalized unit vector.

        Raises:
            ZeroDivisionError: If the vector has zero magnitude (cannot be normalized).
        """
        magnitude = sqrt(self.x ** 2 + self.y ** 2)
        if magnitude != 0:
            return Vector2(self.x / magnitude, self.y / magnitude)
        raise ZeroDivisionError("Cannot normalize a zero-length vector")

    def dot(self, other):
        """
        Computes the dot product between this vector and another vector.

        Args:
            other (Vector2): The other vector for the dot product calculation.

        Returns:
            float: The dot product value.

        Raises:
            TypeError: If the operand type for dot product is not supported.
        """
        if isinstance(other, Vector2):
            return self.x * other.x + self.y * other.y
        raise TypeError("Unsupported operand type for dot product")

    def cross(self, other):
        """
        Computes the cross product between this vector and another vector.

        Args:
            other (Vector2): The other vector for the cross product calculation.

        Returns:
            float: The cross product value.

        Raises:
            TypeError: If the operand type for cross product is not supported.
        """
        if isinstance(other, Vector2):
            return self.x * other.y - self.y * other.x
        raise TypeError("Unsupported operand type for cross product")


Vec2 = Vector2


@dataclass
class Ray:
    """
    Represents a ray in two-dimensional space.

    Attributes:
        pos (Vec2): The position vector of the ray.
        mom (Vec2): The momentum vector of the ray.
    """

    pos: Vec2
    mom: Vec2

    @property
    def q(self):
        """
        Getter for the position vector of the ray.
        """
        return self.pos

    @property
    def p(self):
        """
        Getter for the momentum vector of the ray.
        """
        return self.mom

    def set(self, _pos: Vec2, _mom: Vec2):
        """
        Sets the position and momentum vectors of the ray.

        Args:
            _pos (Vec2): The new position vector.
            _mom (Vec2): The new momentum vector.
        """
        self.pos = _pos
        self.mom = _mom

    def line(self, _x: float):
        """
        Computes the y-coordinate on the ray's line at a given x-coordinate.

        Args:
            _x (float): The x-coordinate.

        Returns:
            float: The corresponding y-coordinate on the ray's line.
        """
        return self.pos.y + (_x - self.pos.x) / self.mom.x * self.mom.y

    def move(self, _dt: float):
        """
        Moves the ray's position along its momentum vector.

        Args:
            _dt (float): The time interval to move the ray.
        """
        self.pos += self.mom * _dt

    def reflect(self, _n: Vec2):
        """
        Reflects the ray's momentum vector based on the given normal vector.

        Args:
            _n (Vec2): The normal vector used for reflection.
        """
        self.mom = self.mom - 2 * self.mom.dot(_n) * _n

    def get_reflected(self, _n: Vec2):
        """
        Calculates the reflected momentum vector based on the given normal vector.

        Args:
            _n (Vec2): The normal vector used for reflection.

        Returns:
            Vec2: The reflected momentum vector.
        """
        return self.mom - 2 * self.mom.dot(_n) * _n


def generate_ray(_pos: Vec2 = Vec2(0., 0.)):
    """
    Generates a random ray starting from the given position.

    Args:
        _pos (Vec2, optional): The starting position vector of the ray. Defaults to Vec2(0., 0.).

    Returns:
        Ray: The generated random ray.
    """
    theta = np.random.uniform(-np.pi, np.pi)
    return Ray(_pos, Vec2(np.cos(theta), np.sin(theta)))


def forward(_ray: Ray, _d: float):
    """
    Checks if the ray is moving forward based on a distance threshold.

    Args:
        _ray (Ray): The ray to check.
        _d (float): The distance threshold.

    Returns:
        bool: True if the ray is moving forward, False otherwise.
    """
    return (_d - _ray.pos.x) / _ray.mom.x > 0


@dataclass
class Quadratic:
    a: float
    b: float
    c: float

    @property
    def mu(self):
        """
        Getter for the midpoint of the quadratic function.
        """
        return -self.b / (2 * self.a)

    @property
    def nu(self):
        """
        Getter for the width of the quadratic function.
        """
        return 1 / (2 * self.a)

    def eval(self, _x: float) -> float:
        """
        Evaluates the quadratic function at a given x-value.

        Args:
            _x (float): The x-value at which to evaluate the quadratic function.

        Returns:
            float: The value of the quadratic function at the given x-value.
        """
        return self.a * _x ** 2 + self.b * _x + self.c

    def diff(self, _x: float) -> float:
        """
        Computes the derivative of the quadratic function at a given x-value.

        Args:
            _x (float): The x-value at which to compute the derivative.

        Returns:
            float: The derivative of the quadratic function at the given x-value.
        """
        return 2 * self.a * _x + self.b

    def discri(self, _qx: float, _qy: float, _px: float, _py: float) -> float:
        """
        Computes the discriminant of the quadratic function intercept for a given ray.

        Args:
            _qx (float): The x-coordinate of the ray's position.
            _qy (float): The y-coordinate of the ray's position.
            _px (float): The x-component of the ray's momentum vector.
            _py (float): The y-component of the ray's momentum vector.

        Returns:
            float: The discriminant of the quadratic function for the given ray.
        """
        return -4 * self.a * self.c * _px ** 2 \
            + 4 * self.a * _px ** 2 * _qy \
            - 4 * self.a * _px * _py * _qx \
            + self.b ** 2 * _px ** 2 \
            - 2 * self.b * _px * _py \
            + _py ** 2

    def normal(self, _x: float) -> Vec2:
        """
        Computes the normal vector to the quadratic function at a given x-value.

        Args:
            _x (float): The x-value at which to compute the normal vector.

        Returns:
            Vec2: The normalized normal vector to the quadratic function at the given x-value.
        """
        return Vec2(np.sign(self.a) * self.diff(_x), -np.sign(self.a)).normalize()

    def find_roots(self, _d: float, _px: float, _py: float) -> Tuple[float, float]:
        """
        Finds the roots of the quadratic equation based on the discriminant and ray parameters.

        Args:
            _d (float): The discriminant of the quadratic equation.
            _px (float): The x-component of the ray's momentum vector.
            _py (float): The y-component of the ray's momentum vector.

        Returns:
            Tuple[float, float]: The roots of the quadratic equation.
        """
        r1 = -(self.b * _px - _py) / (2 * self.a * _px) + sqrt(_d) / (2 * self.a * _px)
        r2 = -(self.b * _px - _py) / (2 * self.a * _px) - sqrt(_d) / (2 * self.a * _px)
        return r1, r2

    def hit(self,
            _qx: float, _qy: float,
            _px: float, _py: float) -> Optional[float]:
        """
        Finds the intersection point of the ray with the quadratic function.

        Args:
            _qx (float): The x-coordinate of the ray's position.
            _qy (float): The y-coordinate of the ray's position.
            _px (float): The x-component of the ray's momentum vector.
            _py (float): The y-component of the ray's momentum vector.

        Returns:
            Optional[float]: The intersection point of the ray with the quadratic function.
                             Returns None if there is no intersection.
        """
        d = self.discri(_qx, _qy, _px, _py)
        if d < 0:
            return None

        roots = self.find_roots(d, _px, _py)
        # Sort roots via distance from ray origin
        roots = sorted(roots, key=lambda x: abs(x - _qx))
        for root in roots:
            if not isclose(_qx, root, abs_tol=1e-5) and (root - _qx) / _px > 0:
                return root
        return None


@dataclass
class QuadraticFromRoots:
    """
    Represents a quadratic function of the form a(x - xmin)(x - xmax).

    Args:
        a (float): The coefficient of the quadratic term.
        xmin (float): The minimum x-value of the quadratic function.
        xmax (float): The maximum x-value of the quadratic function.
        tag (int, optional): An optional tag for the quadratic function. Defaults to 0.
    """

    a: float
    xmin: float
    xmax: float
    tag: int = 0

    @property
    def mu(self):
        """
        Getter for the midpoint of the quadratic function.
        """
        return (self.xmin + self.xmax) / 2

    @property
    def nu(self):
        """
        Getter for the width of the quadratic function.
        """
        return (self.xmin - self.xmax) / 2

    def eval(self, _x: float) -> float:
        """
        Evaluates the quadratic function at a given x-value.

        Args:
            _x (float): The x-value at which to evaluate the quadratic function.

        Returns:
            float: The value of the quadratic function at the given x-value.
        """
        return self.a * (_x - self.xmin) * (_x - self.xmax)

    def diff(self, _x: float) -> float:
        """
        Computes the derivative of the quadratic function at a given x-value.

        Args:
            _x (float): The x-value at which to compute the derivative.

        Returns:
            float: The derivative of the quadratic function at the given x-value.
        """
        return self.a * (2 * _x - self.xmin - self.xmax)

    def discri(self, _qx: float, _qy: float, _px: float, _py: float) -> float:
        """
        Computes the discriminant of the quadratic intercept function for a given ray.

        Args:
            _qx (float): The x-coordinate of the ray's position.
            _qy (float): The y-coordinate of the ray's position.
            _px (float): The x-component of the ray's momentum vector.
            _py (float): The y-component of the ray's momentum vector.

        Returns:
            float: The discriminant of the quadratic function for the given ray.
        """
        return self.a ** 2 * _px ** 2 * self.xmax ** 2 \
            - 2 * self.a ** 2 * _px ** 2 * self.xmax * self.xmin \
            + self.a ** 2 * _px ** 2 * self.xmin ** 2 \
            + 4 * self.a * _px ** 2 * _qy \
            - 4 * self.a * _px * _py * _qx \
            + 2 * self.a * _px * _py * self.xmax \
            + 2 * self.a * _px * _py * self.xmin \
            + _py ** 2

    def normal(self, _x: float) -> Vec2:
        """
        Computes the normal vector to the quadratic function at a given x-value.

        Args:
            _x (float): The x-value at which to compute the normal vector.

        Returns:
            Vec2: The normalized normal vector to the quadratic function at the given x-value.
        """
        return Vec2(-np.sign(self.a) * self.diff(_x), np.sign(self.a)).normalize()

    def find_roots(self, _d: float, _px: float, _py: float) -> Tuple[float, float]:
        """
        Finds the roots of the quadratic equation based on the discriminant and ray parameters.

        Args:
            _d (float): The discriminant of the quadratic equation.
            _px (float): The x-component of the ray's momentum vector.
            _py (float): The y-component of the ray's momentum vector.

        Returns:
            Tuple[float, float]: The roots of the quadratic equation.
        """
        r1 = (self.a * _px * self.xmax + self.a * _px * self.xmin + _py) / (2 * self.a * _px) \
             + sqrt(_d) / (2 * self.a * _px)
        r2 = (self.a * _px * self.xmax + self.a * _px * self.xmin + _py) / (2 * self.a * _px) \
             - sqrt(_d) / (2 * self.a * _px)
        return r1, r2

    def hit(self,
            _qx: float, _qy: float,
            _px: float, _py: float) -> Optional[float]:
        """
        Finds the intersection point of the ray with the quadratic function.

        Args:
            _qx (float): The x-coordinate of the ray's position.
            _qy (float): The y-coordinate of the ray's position.
            _px (float): The x-component of the ray's momentum vector.
            _py (float): The y-component of the ray's momentum vector.

        Returns:
            Optional[float]: The intersection point of the ray with the quadratic function.
                             Returns None if there is no intersection.
        """
        d = self.discri(_qx, _qy, _px, _py)
        if d < 0:
            return None
        roots = self.find_roots(d, _px, _py)
        for root in roots:
            if not isclose(_qx, root, abs_tol=1e-5) and (root - _qx) / _px > 0:
                return root
        return None


def within_bounds(_root: float, _min: float, _max: float) -> bool:
    """
    Checks if a value is within the specified bounds.

    Args:
        _root (float): The value to check.
        _min (float): The minimum bound.
        _max (float): The maximum bound.

    Returns:
        bool: True if the value is within the specified bounds, False otherwise.
    """
    return _max > _root > _min


def reflected_ray(_root: float, _ray: Ray, _quadratic: QuadraticFromRoots) -> Ray:
    """
    Generates a new ray by reflecting an existing ray off a quadratic surface.

    Args:
        _root (float): The root representing the intersection point of the ray with the quadratic surface.
        _ray (Ray): The original ray.
        _quadratic (QuadraticFromRoots): The quadratic surface.

    Returns:
        Ray: The reflected ray.
    """
    _n = _quadratic.normal(_root)
    _mom = _ray.get_reflected(_n)
    _pos = Vec2(_root, _quadratic.eval(_root))
    return Ray(_pos, _mom)


def trace_ray(_ray: Ray,
              _bounds: List[QuadraticFromRoots],
              _da_min: float, _da_max: float, _x_screen: float,
              _rays: List[Ray] = None,
              verbose: bool = False,
              _max_iterations: int = 100) -> Optional[float]:
    """
    Traces a ray through a series of quadratic bounds until it reaches the screen or exits the bounds.

    Args:
        _ray (Ray): The initial ray.
        _bounds (List[QuadraticFromRoots]): A list of quadratic bounds.
        _da_min (float): The minimum value of the design area along the x-axis.
        _da_max (float): The maximum value of the design area along the x-axis.
        _x_screen (float): The x-coordinate of the screen.
        _rays (List[Ray], optional): A list to store the traced rays. Defaults to None.
        verbose (bool, optional): Whether to print verbose output. Defaults to False.
        _max_iterations (int, optional): The maximum number of iterations. Defaults to 100.

    Returns:
        Optional[float]: The y-coordinate where the ray crosses the screen, or None if it exits the bounds.

    """
    iteration = 0
    while iteration < _max_iterations:
        ref_rays = []
        cross_bounds = False

        for bound in _bounds:
            root = bound.hit(_ray.pos.x, _ray.pos.y, _ray.mom.x, _ray.mom.y)
            if root is not None and within_bounds(root, _da_min, _da_max):
                ref_rays.append(reflected_ray(root, _ray, bound))
                cross_bounds = True

        if len(ref_rays) > 1:
            # keep the ray with the smallest distance from _ray
            ref_rays.sort(key=lambda x: (_ray.pos.x - x.pos.x) ** 2 + (_ray.pos.y - x.pos.y) ** 2)

        if cross_bounds:
            _ray = ref_rays[0]
            if _rays is not None:
                _rays.append(_ray)
        else:
            if forward(_ray, _x_screen):
                t = (_x_screen - _ray.pos.x) / _ray.mom.x
                y = _ray.pos.y + t * _ray.mom.y
                if verbose:
                    print(f'Screen crossed at y={y}cm')
                return y
            break

        iteration += 1
        if verbose:
            print(_ray)


def draw_rays(rays: List[Ray]) -> None:
    """
    Draws a series of rays on a plot.

    Args:
        rays (List[Ray]): A list of rays to be drawn.

    Returns:
        None
    """
    for first, second in zip(rays, rays[1:]):
        x = np.linspace(first.pos.x, second.pos.x, 100)
        y = first.line(x)
        plt.plot(x, y, color='k', linewidth=0.5)
        plt.arrow(first.pos.x, first.pos.y, first.mom.x, first.mom.y,
                  head_width=0.2, head_length=0.2, fc='k', ec='k')
        # If last ray change color to red
        if second == rays[-1]:
            plt.arrow(second.pos.x, second.pos.y, second.mom.x, second.mom.y,
                      head_width=0.2, head_length=0.2, fc='r', ec='r')
    plt.show()


def draw(_rays: List[Ray],
         _bounds: List[QuadraticFromRoots],
         _da_min: float, _da_max: float) -> None:
    """
    Draws rays and quadratic bounds on a plot.

    Args:
        _rays (List[Ray]): A list of rays to be drawn.
        _bounds (List[QuadraticFromRoots]): A list of quadratic bounds to be drawn.
        _da_min (float): The minimum x-value for the plot.
        _da_max (float): The maximum x-value for the plot.

    Returns:
        None
    """
    x = np.linspace(_da_min, _da_max, 100)
    for bound in _bounds:
        y = bound.eval(x)
        plt.plot(x, y)
    # Plot
    plt.axis('equal')
    draw_rays(_rays)
