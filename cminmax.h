#pragma once

int inline constexpr cmin(int a, int b) { return a > b ? b : a; }
int inline constexpr cmax(int a, int b) { return a > b ? a : b; }
