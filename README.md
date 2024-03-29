# HPC_HW
Source code for university course "High-Performance Computing" homeworks
Домашние работы по курсу "Высокопроизводительные вычисления", магистратура, 1 курс, 1 семестр (осенний семестр 2022).

## ДЗ1
Постановка:
- Реализовать умножение матриц DGEMM из пакета BLAS с использованием технологии OpenMP.
- Привести анализ сильной/слабой масштабируемости параллельной реализации на суперкомпьютере Харизма.
- Реализовать оптимизированную под узлы суперкомпьютера Харизма параллельную реализацию DGEMM, проанализировать характеристики ее сильной/слабой масштабируемости.
- Отдельное доп. задание для всех: реализовать вычисление суммы ряда с использованием pthreads и поддержкой произвольного количества потоков.
- <b>Отдельное доп. задание для меня: реализовать умножение матриц с помощью AVX-инструкций.</b>

Требования к выполнению:
- Тип данных double.
- Column-major order для двух исходных матриц и результата.
- Реализация на языке С.
- Оформление кода в соответствии с <i>Linux Kernel Codestyle</i>.
- Комментарии кода должны приводиться на английском языке и объяснять неочевидные моменты, а не дублировать написанный код

## ДЗ2
Постановка:
1. Реализовать перемножение квадратных матриц с использованием глобальной памяти на CUDA.
2. Ускорить передачу матрицы из CPU в GPU за счет использования Pinned памяти.
3. Ускорить передачу матрицы из CPU в GPU за счет использования CUDA потоков.
4. Написать перемножение матриц с использованием разделяемой памяти.
5. Для каждого из предыдущих пунктов представить сравнение времени работы программы при разных размерностях матриц

## ДЗ3
Постановка:
1. Решить одномерное однородное уравнение теплопроводности с использование средств распараллеливания MPI.
2. Проверить работу параллельной реализации с помощью сравнения с точным решением на основе суммы ряда.
3. Сравнить время работы для разного кол-ва процессов, точек на стержне и шага по времени.
4. Использовать коллективные операции для рассылки начального распределения температур и сбора итоговых результатов.
