#pragma once

#include <chrono>

using namespace std::chrono;



class Stopwatch{
private:
	bool _isRunning = false;
	duration<double> _dur = 0ms;
	time_point<system_clock, duration<long, std::ratio<1, 1000000000>>>  _start;

public:
	void start(){
		if (_isRunning) return;
		_start = high_resolution_clock::now();
		_isRunning = true;
	}
	
	void stop(){
		if (!_isRunning) return;
		
		_dur += high_resolution_clock::now() - _start;
		_isRunning = false;
	}
	
	void reset(){
		_dur = 0ms;
	}
	
	duration<double> getDuration(){
		return _dur;
	}
};
