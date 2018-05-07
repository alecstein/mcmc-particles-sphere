# Simulating interacting particles on a sphere

I've used the Metropolis-Hastings algorithm to calculate the radial distribution function for particles interacting on the surface of a sphere.

## Methods

I work using cartesian coordinates, since these simplify both the legibility of the code and the math. It's a little unweildy to code a random walk around the sphere. Instead, each particle takes a random jump in any direction (meaning it is allowed to pop off the sphere) and then is projected back onto the sphere. This move allows the particle to move uniformly in any direction.
