# `madrona_physics`
Test bed for new Madrona physics features.

## TODO:
- Split up allocation of CV registers based on "dynamic"-ness
- Determine which registers from allocateScratch actually need `max_num_comps` elements.
- Implement sparse matrix representations for contact and equality jacobians.
- Try parallelizing compositeRigidBody - each thread does one component of the mass matrix
- Remove need to get body group when querying body dof offset.
- Get rid of shit in line 3351 cvhpyscis.cpp
- REMEMBER THAT THERE IS ONLY ONE JOINT LIMIT PER ROW
- Use atomics for the the column version of the equality jacobian

- We can parallelize some parts of solveM
- Attach limit to each DofObjectArchetype - useful for prepareSolver GPU backend
