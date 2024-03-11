#pragma once
//#include <cmath>

const double PI = 3.1415926535897932;

// Длина стержня (начинается в точке 0.0)
const double ROD_LENGTH = 1.0;
// Целевое значение времени
const double TIME_LENGTH = 1e-01;
// Начальная температура стержня
const double ROD_INITIAL_TEMP = 1.0;
// Поддерживаемая температура среды
const double ENVIROMENT_TEMP = 0.0;
// К-т температуропроводности
const double CONDUCTIVITY_RATIO = 1.0;
// Шаг по пространству вдоль стержня
const double SPATIAL_STEP = 2e-02;
// Шаг по времени
const double TIME_STEP = 2e-04;

// Общее число подсчитываемых точек
const int N_POINTS = static_cast<int>(ROD_LENGTH / SPATIAL_STEP) + 1;

// Общее число шагов по времени
const int TIME_STEPS = TIME_LENGTH / TIME_STEP + 1;

// Коэффициент, используемый для конечно-разностной схемы
const double EQUATION_RATIO = (CONDUCTIVITY_RATIO * TIME_STEP) / (SPATIAL_STEP * SPATIAL_STEP);


// Базовый номер для сообщения, в котором содержатся сведения о крайней левой точке процесса
const int MPI_SEND_LEFT_EDGE_DATA_BASE_TAG = 10;
// Базовый номер для сообщения, в котором содержатся сведения о крайней правой точке процесса
const int MPI_SEND_RIGHT_EDGE_DATA_BASE_TAG = 11;