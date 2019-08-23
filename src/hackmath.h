//
// Created by xabab on 14.08.19.
//

#ifndef NEURAL_SURVIVAL_SIM_CPP_HACKMATH_H
#define NEURAL_SURVIVAL_SIM_CPP_HACKMATH_H


float sqrt3(const float x) {
	union {
		int i;
		float x;
	} u{};
	
	u.x = x;
	u.i = (1 << 29) + (u.i >> 1) - (1 << 22);
	return u.x;
}

#define SQRT_MAGIC_F 0x5f3759df
float sqrt2(const float x)
{
	const float xhalf = 0.5f*x;
	
	union // get bits for floating value
	{
		float x;
		int i;
	} u{};
	u.x = x;
	u.i = SQRT_MAGIC_F - (u.i >> 1);  // gives initial guess y0
	return x*u.x*(1.5f - xhalf*u.x*u.x);// Newton step, repeating increases accuracy
}


#endif //NEURAL_SURVIVAL_SIM_CPP_HACKMATH_H