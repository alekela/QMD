import numpy as np
from .data import field

def dummy(elements):
	pass

class RK1:	
	def __init__(self, solver, derivative = {}, models=dummy, CFL = 0.5, before=dummy, after=dummy):
		self.solver = solver
		self.models = models
		self.derivative = derivative
		self.CFL = CFL
		self.before = before
		self.after = after

	def get_step(self, elements):
		new_dt = 1e+15

		v = np.zeros(elements.size)
		c = np.zeros(elements.size)
		vs = np.zeros(elements.size)
		if field.velocity in elements.fields:
			v[:] = np.abs(elements[field.velocity])
		if field.soundSpeed in elements.fields:
			c[:] = elements[field.soundSpeed]
		if field.strainRate in elements.fields:
			vs[:] = np.abs(elements[field.size] * elements[field.strainRate])
		new_dt = np.amin(self.CFL*elements[field.size]/(v + c + vs))
		return new_dt

	def make_step(self, elements, dt):
		for f in self.derivative:
			elements[f] += dt * elements[self.derivative[f]]
		
	def update(self, elements):
		time = 0.0
		while True:
			dt = self.get_step(elements)
			self.before(elements)
			self.solver(elements)
			self.make_step(elements, dt)
			self.after(elements)
			self.models(elements)
			time += dt
			yield time