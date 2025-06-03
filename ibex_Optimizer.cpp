// I B E X
// File : ibex_Optimizer.cpp
// Author : Gilles Chabert, Bertrand Neveu
// Copyright : IMT Atlantique (France)
// License : See the LICENSE file
// Created : May 14, 2012
// Last Update : Feb 13, 2025
//============================================================================
#include "ibex_Optimizer.h"
#include "ibex_Timer.h"
#include "ibex_Function.h"
#include "ibex_NoBisectableVariableException.h"
#include "ibex_BxpOptimData.h"
#include "ibex_CovOptimData.h"
#include <float.h>
#include <stdlib.h>
#include <iomanip>
#include <fstream>	 // Para ofstream/*
#include <string>	 // Para std::to_string
#include <algorithm> // Para std::min (si lo necesitas en otra parte)
using namespace std;
// ——————————————————————————————————————————————————————————————————————
// Código de clustering embebido (k-means)
#include <vector>
#include <random>
#include <limits>
// Representa el centro de cada caja (dim = n+1)
using Point = std::vector<double>;
// Resultado de k-means: etiqueta por punto y nº de clústeres
struct ClusterResult
{
	std::vector<int> labels;
	int n_clusters;
};
/// Versión simple de k-means (50 iters, semilla fija para reproducibilidad)
static ClusterResult kmeans(const std::vector<Point> &data, int k)
{
	int n = data.size();
	int dim = data[0].size();
	std::mt19937 gen(0);
	std::uniform_int_distribution<> dis(0, n - 1);
	// 1) Inicializar centroides
	std::vector<Point> centroids(k, Point(dim));
	for (int i = 0; i < k; ++i)
		centroids[i] = data[dis(gen)];
	std::vector<int> labels(n, 0);
	bool changed = true;
	for (int iter = 0; iter < 50 && changed; ++iter)
	{
		changed = false;
		// 2) Asignación
		for (int i = 0; i < n; ++i)
		{
			double best = std::numeric_limits<double>::infinity();
			int bi = 0;
			for (int c = 0; c < k; ++c)
			{
				double d = 0;
				for (int d0 = 0; d0 < dim; ++d0)
				{
					double diff = data[i][d0] - centroids[c][d0];
					d += diff * diff;
				}
				if (d < best)
				{
					best = d;
					bi = c;
				}
			}
			if (labels[i] != bi)
			{
				labels[i] = bi;
				changed = true;
			}
		}
		// 3) Re-cálculo de centroides
		std::vector<Point> sum(k, Point(dim, 0.0));
		std::vector<int> cnt(k, 0);
		for (int i = 0; i < n; ++i)
		{
			cnt[labels[i]]++;
			for (int d0 = 0; d0 < dim; ++d0)
				sum[labels[i]][d0] += data[i][d0];
		}
		for (int c = 0; c < k; ++c)
		{
			if (cnt[c] > 0)
				for (int d0 = 0; d0 < dim; ++d0)
					centroids[c][d0] = sum[c][d0] / cnt[c];
			else
				centroids[c] = data[dis(gen)];
		}
	}
	return ClusterResult{labels, k};
}
// ──────────────────────────────────────────────────────────────────────────────
#include <cmath>
const int DBSCAN_NOISE = -1;
const int DBSCAN_UNCLASSIFIED = -2; // O cualquier otro valor negativo distinto de -1
// Estructura para el resultado de DBSCAN
struct DbscanClusterResult
{
	std::vector<int> labels; // labels[i] = ID del clúster (0 a num_clusters-1) o DBSCAN_NOISE
	int num_clusters;		 // Número de clústeres reales encontrados (excluyendo el ruido)
};
// Función para calcular la distancia euclidiana entre dos puntos
static double calculate_euclidean_distance_for_dbscan(const Point &p1, const Point &p2)
{
	double sum_sq_diff = 0.0;
	// Asumimos que p1 y p2 tienen la misma dimensionalidad
	for (size_t i = 0; i < p1.size(); ++i)
	{
		double diff = p1[i] - p2[i];
		sum_sq_diff += diff * diff;
	}
	return std::sqrt(sum_sq_diff);
}
// Encuentra todos los puntos dentro de la distancia 'eps' del punto 'point_idx'
// Devuelve los índices de estos puntos vecinos.
static std::vector<int> region_query_for_dbscan(int point_idx, double eps, const std::vector<Point> &data)
{
	std::vector<int> neighbors;
	for (size_t i = 0; i < data.size(); ++i)
	{
		if (calculate_euclidean_distance_for_dbscan(data[point_idx], data[i]) <= eps)
		{
			neighbors.push_back(i);
		}
	}
	return neighbors;
}
// Función principal de DBSCAN
static DbscanClusterResult dbscan(const std::vector<Point> &data, double eps, int min_pts)
{
	int n = data.size();
	DbscanClusterResult result;
	result.labels.assign(n, DBSCAN_UNCLASSIFIED); // Inicialmente todos no clasificados
	result.num_clusters = 0;
	if (n == 0 || min_pts <= 0)
	{ // Casos base o inválidos
		if (n > 0)
			result.labels.assign(n, DBSCAN_NOISE); // Marcar todos como ruido si min_pts es inválido
		return result;
	}
	for (int i = 0; i < n; ++i)
	{
		if (result.labels[i] != DBSCAN_UNCLASSIFIED)
		{ // Ya procesado
			continue;
		}
		std::vector<int> neighbors = region_query_for_dbscan(i, eps, data);
		if (neighbors.size() < (size_t)min_pts)
		{ // Densidad insuficiente, marcar como ruido
			result.labels[i] = DBSCAN_NOISE;
			continue;
		}
		// Es un core point, iniciar un nuevo clúster
		int current_cluster_id = result.num_clusters;
		result.num_clusters++;
		result.labels[i] = current_cluster_id; // Asignar punto actual al nuevo clúster
		// Expandir el clúster desde este core point
		std::vector<int> seed_set = neighbors; // Usar 'neighbors' como la semilla inicial
		size_t seed_set_current_idx = 0;
		while (seed_set_current_idx < seed_set.size())
		{
			int q_idx = seed_set[seed_set_current_idx];
			seed_set_current_idx++;
			// Si q_idx era ruido y ahora es alcanzable, se convierte en border point del clúster actual
			if (result.labels[q_idx] == DBSCAN_NOISE)
			{
				result.labels[q_idx] = current_cluster_id;
			}
			// Si q_idx no ha sido clasificado aún, asígnale este clúster
			if (result.labels[q_idx] == DBSCAN_UNCLASSIFIED)
			{
				result.labels[q_idx] = current_cluster_id;
				// Si q_idx también es un core point, añade sus vecinos a la semilla
				std::vector<int> q_neighbors = region_query_for_dbscan(q_idx, eps, data);
				if (q_neighbors.size() >= (size_t)min_pts)
				{
					for (int neighbor_of_q_idx : q_neighbors)
					{
						// Añadir solo si no clasificado o ruido (podría unirse al clúster)
						if (result.labels[neighbor_of_q_idx] == DBSCAN_UNCLASSIFIED ||
							result.labels[neighbor_of_q_idx] == DBSCAN_NOISE)
						{
							// Opcional: verificar si ya está en seed_set para evitar duplicados y reprocesamiento
							// bool found_in_seed = false;
							// for(size_t s_idx=0; s_idx < seed_set.size(); ++s_idx) if(seed_set[s_idx] == neighbor_of_q_idx) {found_in_seed=true; break;}
							// if(!found_in_seed) seed_set.push_back(neighbor_of_q_idx);
							seed_set.push_back(neighbor_of_q_idx); // Más simple, la lógica de visitado/clasificado maneja redundancias
						}
					}
				}
			}
		}
	}
	return result;
}
// --- Fin del Código DBSCAN ---    


// Nueva función para estimar un eps dinámico
static double estimate_dynamic_eps(const std::vector<Point> &data,
								   int min_pts_for_context,		// Para asegurar que hay suficientes puntos para una estadística
								   double fraction_of_avg_dist, // Qué fracción de la distancia promedio usar
								   double default_eps_if_too_few_points = 0.1)
{ // Fallback
	size_t n_points = data.size();
	// Si hay muy pocos puntos para una estimación robusta, o para satisfacer min_pts,
	// es mejor usar un eps por defecto o no intentar el clustering.
	if (n_points < 2 || n_points < (size_t)min_pts_for_context)
	{
		return default_eps_if_too_few_points;
	}
	int dim = 0;
	if (n_points > 0)
	{ // Asegurar que data no está vacío antes de acceder a data[0]
		dim = data[0].size();
	}
	if (dim == 0)
		return default_eps_if_too_few_points; // Puntos sin dimensión
	Point mean_point(dim, 0.0);
	// 1. Calcular el centroide (punto promedio) de todos los 'centers'
	for (const auto &p : data)
	{
		for (int d = 0; d < dim; ++d)
		{
			mean_point[d] += p[d];
		}
	}
	for (int d = 0; d < dim; ++d)
	{
		mean_point[d] /= n_points;
	}
	// 2. Calcular la distancia promedio de todos los puntos a este centroide
	double total_distance_to_mean = 0.0;
	for (const auto &p : data)
	{
		total_distance_to_mean += calculate_euclidean_distance_for_dbscan(p, mean_point); // Tu función de distancia
	}
	double avg_distance_to_mean = (n_points > 0) ? (total_distance_to_mean / n_points) : 0.0;
	// 3. Establecer eps como una fracción de esta distancia promedio
	// 'fraction_of_avg_dist' es un nuevo parámetro que podrías necesitar ajustar (ej. 0.1 a 0.5)
	double dynamic_eps_candidate = avg_distance_to_mean * fraction_of_avg_dist;
	// Asegurar que eps no sea cero si hay dispersión, o darle un valor mínimo muy pequeño
	if (dynamic_eps_candidate == 0.0)
	{
		if (avg_distance_to_mean > 1e-9)
		{														// Si había algo de dispersión pero el cálculo dio 0
			dynamic_eps_candidate = avg_distance_to_mean * 0.1; // Una fracción pequeña
		}
		else
		{								  // Todos los puntos son idénticos o están en el centroide
			dynamic_eps_candidate = 1e-3; // Un valor pequeño por defecto para evitar eps=0
		}
	}
	return dynamic_eps_candidate;
}


// KMEDOIDS
static ClusterResult kmedoids(const std::vector<Point> &data, int k, int max_iters = 50)
{
	int n = data.size();
	ClusterResult result;
	if (n == 0 || k <= 0 || k > n)
	{
		result.labels.assign(n, -1); // O alguna etiqueta por defecto para error/sin clúster
		result.n_clusters = 0;
		if (n > 0 && k == 1)
		{ // Caso trivial: un solo clúster con todo
			result.labels.assign(n, 0);
			result.n_clusters = 1;
		}
		return result;
	}
	int dim = data[0].size();
	result.labels.assign(n, 0);
	result.n_clusters = k;				// K-Medoids siempre intenta encontrar k clústeres
	std::vector<int> medoid_indices(k); // Índices en 'data' de los medoides actuales
	std::vector<Point> current_medoids(k, Point(dim));
	// 1. Inicialización: Seleccionar k puntos distintos aleatoriamente como medoides iniciales
	std::mt19937 gen(0); // Misma semilla que k-means para reproducibilidad
	std::vector<int> point_indices(n);
	std::iota(point_indices.begin(), point_indices.end(), 0); // 0, 1, ..., n-1
	std::shuffle(point_indices.begin(), point_indices.end(), gen);
	for (int i = 0; i < k; ++i)
	{
		medoid_indices[i] = point_indices[i];
		current_medoids[i] = data[medoid_indices[i]];
	}
	bool changed_in_iteration = true;
	for (int iter = 0; iter < max_iters && changed_in_iteration; ++iter)
	{
		changed_in_iteration = false;
		// 2. Paso de Asignación: Asignar cada punto al medoide más cercano
		for (int i = 0; i < n; ++i)
		{
			double min_dist = std::numeric_limits<double>::infinity();
			int best_medoid_cluster_idx = 0; // Índice del clúster (0 a k-1)
			for (int ki = 0; ki < k; ++ki)
			{ // ki es el índice del clúster
				double dist = calculate_euclidean_distance_for_dbscan(data[i], current_medoids[ki]);
				if (dist < min_dist)
				{
					min_dist = dist;
					best_medoid_cluster_idx = ki;
				}
			}
			if (result.labels[i] != best_medoid_cluster_idx)
			{
				result.labels[i] = best_medoid_cluster_idx;
				// No marcamos changed_in_iteration aquí, se basa en si los medoides cambian
			}
		}
		// 3. Paso de Actualización: Para cada clúster, encontrar el punto dentro de él
		// que minimiza la suma de distancias a los demás puntos de ESE MISMO clúster.
		// Este punto se convierte en el nuevo medoide.
		std::vector<int> new_medoid_indices(k);
		for (int ki = 0; ki < k; ++ki)
		{													 // Para cada clúster ki
			std::vector<int> points_in_this_cluster_indices; // Índices (en 'data') de los puntos en este clúster
			for (int i = 0; i < n; ++i)
			{
				if (result.labels[i] == ki)
				{
					points_in_this_cluster_indices.push_back(i);
				}
			}
			if (points_in_this_cluster_indices.empty())
			{
				// Clúster vacío: re-inicializar su medoide aleatoriamente
				// (debe ser un punto no ya elegido como medoide por otro clúster si es posible)
				// Esta es una situación que hay que manejar, por simplicidad aquí se podría
				// mantener el medoide anterior o elegir uno al azar de los no medoides.
				// Por ahora, mantenemos el anterior si el clúster se vació.
				new_medoid_indices[ki] = medoid_indices[ki]; // Mantener el medoide si el clúster se vació
				// Si 'medoid_indices[ki]' no era válido o quieres forzar una re-selección:
				// bool unique_new_medoid;
				// do {
				// unique_new_medoid = true;
				// new_medoid_indices[ki] = point_indices[gen() % n]; // Elegir uno al azar
				// for(int prev_ki = 0; prev_ki < ki; ++prev_ki) { // Evitar duplicados con otros nuevos medoides
				// if (new_medoid_indices[ki] == new_medoid_indices[prev_ki]) unique_new_medoid = false;
				// }
				// } while(!unique_new_medoid);
				continue; // Saltar al siguiente clúster
			}
			double min_sum_dist_for_this_cluster = std::numeric_limits<double>::infinity();
			int best_new_medoid_data_idx = points_in_this_cluster_indices[0]; // Candidato inicial
			// Iterar sobre cada punto 'p_candidate_idx' del clúster como posible nuevo medoide
			for (int p_candidate_idx : points_in_this_cluster_indices)
			{
				double current_sum_dist_for_candidate = 0;
				// Calcular la suma de distancias desde este candidato a todos los demás puntos del clúster
				for (int p_other_idx : points_in_this_cluster_indices)
				{
					current_sum_dist_for_candidate +=
						calculate_euclidean_distance_for_dbscan(data[p_candidate_idx], data[p_other_idx]);
				}
				if (current_sum_dist_for_candidate < min_sum_dist_for_this_cluster)
				{
					min_sum_dist_for_this_cluster = current_sum_dist_for_candidate;
					best_new_medoid_data_idx = p_candidate_idx;
				}
			}
			new_medoid_indices[ki] = best_new_medoid_data_idx;
		}
		// Comprobar si los medoides cambiaron
		for (int ki = 0; ki < k; ++ki)
		{
			if (medoid_indices[ki] != new_medoid_indices[ki])
			{
				changed_in_iteration = true; // ¡Sí cambiaron!
				break;
			}
		}
		// Actualizar medoides para la siguiente iteración
		if (changed_in_iteration)
		{
			medoid_indices = new_medoid_indices;
			for (int ki = 0; ki < k; ++ki)
			{
				current_medoids[ki] = data[medoid_indices[ki]];
			}
		}
		else
		{ // No hubo cambios en los medoides, convergencia.
			break;
		}
	}
	// La asignación final (labels) ya se hizo en la última iteración.
	return result;
}
// --- Fin del Código K-Medoids ---

#include <iostream> // Para std::cout, std::endl si los usas directamente
namespace {

double calculate_box_volume(const ibex::IntervalVector& box) {
    if (box.is_empty()) {
        return 0.0;
    }
    if (box.size() == 0) { // Caja 0-dimensional
        return 1.0; // Convencionalmente, el volumen de un punto (o el producto vacío) es 1.
                    // Si prefieres 0.0 para una caja sin dimensiones, cámbialo.
    }

    double volume = 1.0;
    bool has_infinite_dimension = false;

    for (int i = 0; i < box.size(); ++i) {
        const ibex::Interval& component = box[i];
        
        if (component.is_empty()) { // Si cualquier componente es vacío, el volumen total es 0
            return 0.0;
        }

        double diam = component.diam();

        if (diam < 0) { // Diámetro negativo implica vacío para Ibex
            return 0.0;
        }

		if (diam == std::numeric_limits<double>::infinity()) {
			has_infinite_dimension = true;
		} else if (diam == 0.0) {
            // Si cualquier dimensión finita tiene diámetro 0, el volumen total es 0.
            // Esto tiene prioridad sobre una dimensión infinita (ej. una línea en un plano tiene volumen 0).
            return 0.0;
        } else {
            // Solo multiplica diámetros finitos y no cero por ahora.
            volume *= diam; 
            // Comprobación temprana de overflow a infinito si el volumen ya es enorme
            if (volume == std::numeric_limits<double>::infinity()) break; 
        }
    }

    // Si hubo una dimensión infinita Y ninguna dimensión con diámetro cero, el volumen es infinito.
    if (has_infinite_dimension) {
        return std::numeric_limits<double>::infinity();
    }

    return volume;
}

} // fin del namespace anónimo

namespace ibex
{
	/*
	 * TODO: redundant with ExtendedSystem.
	 */
	void Optimizer::write_ext_box(const IntervalVector &box, IntervalVector &ext_box)
	{
		int i2 = 0;
		for (int i = 0; i < n; i++, i2++)
		{
			if (i2 == goal_var)
				i2++; // skip goal variable
			ext_box[i2] = box[i];
		}
	}
	void Optimizer::read_ext_box(const IntervalVector &ext_box, IntervalVector &box)
	{
		int i2 = 0;
		for (int i = 0; i < n; i++, i2++)
		{
			if (i2 == goal_var)
				i2++; // skip goal variable
			box[i] = ext_box[i2];
		}
	}
	Optimizer::Optimizer(int n, Ctc &ctc, Bsc &bsc, LoupFinder &finder,
						 CellBufferOptim &buffer,
						 int goal_var, double eps_x, double rel_eps_f, double abs_eps_f,
						 bool enable_statistics) : n(n), goal_var(goal_var),
												   ctc(ctc), bsc(bsc), loup_finder(finder), buffer(buffer),
												   eps_x(n, eps_x), rel_eps_f(rel_eps_f), abs_eps_f(abs_eps_f),
												   trace(0), timeout(-1), extended_COV(true), anticipated_upper_bounding(true),
												   status(SUCCESS),
												   uplo(NEG_INFINITY), uplo_of_epsboxes(POS_INFINITY), loup(POS_INFINITY),
												   loup_point(IntervalVector::empty(n)), initial_loup(POS_INFINITY), loup_changed(false),
												   time(0), nb_cells(0), cov(NULL), clustering_params()
	{
		if (trace)
			cout.precision(12);

		// Inicialización del control de reinicios

		clustering_params.choice = ClusteringParams::Algorithm::DBSCAN;

		restart_threshold = 1000;
		node_threshold = 100000;

		// KMEANS
		stagnation_counter = 0;
		clustering_params.k = 2000; // Tu valor anterior para k-means

		// KMEDOIDS
		clustering_params.kmedoids_max_iters = 50;

		//---------------------------------------------------------------------

		// DBSCAN
		clustering_params.eps = 0.1;  // VALOR DE EJEMPLO - MUY SENSIBLE A TUS DATOS
		clustering_params.minPts = 2; // clustering_params.minPts = this->n + 1; // VALOR DE EJEMPLO (ej. 2*dimensión o un valor fijo pequeño)

		// eps dinamico
		clustering_params.use_dynamic_eps = true;	  // ¡Actívalo para probar!
		clustering_params.dynamic_eps_fraction = 0.09; // Fracción a probar (0.1 a 0.5 es un rango inicial)
		clustering_params.dynamic_eps_fallback = 0.1; // Fallback para eps si hay pocos puntos


		if (enable_statistics)
		{
			statistics = new Statistics();
			// TODO: enable statistics for missing operators (cell buffer)
			bsc.enable_statistics(*statistics, "Bsc");
			ctc.enable_statistics(*statistics, "Ctc");
			loup_finder.enable_statistics(*statistics, "LoupFinder");
		}
		else
			statistics = NULL;
	}
	Optimizer::Optimizer(OptimizerConfig &config) : Optimizer(
														config.nb_var(),
														config.get_ctc(),
														config.get_bsc(),
														config.get_loup_finder(),
														config.get_cell_buffer(),
														config.goal_var(),
														OptimizerConfig::default_eps_x, // tmp, see below
														config.get_rel_eps_f(),
														config.get_abs_eps_f(),
														config.with_statistics())
	{
		(Vector &)eps_x = config.get_eps_x();
		trace = config.get_trace();
		timeout = config.get_timeout();
		extended_COV = config.with_extended_cov();
		anticipated_upper_bounding = config.with_anticipated_upper_bounding();
	}
	Optimizer::~Optimizer()
	{
		if (cov)
			delete cov;
		if (statistics)
			delete statistics;
	}
	// compute the value ymax (decreasing the loup with the precision)
	// the heap and the current box are contracted with y <= ymax
	double Optimizer::compute_ymax()
	{
		if (anticipated_upper_bounding)
		{
			// double ymax = loup - rel_eps_f*fabs(loup); ---> wrong :the relative precision must be correct for ymax (not loup)
			double ymax = loup > 0 ? 1 / (1 + rel_eps_f) * loup
								   : 1 / (1 - rel_eps_f) * loup;
			if (loup - abs_eps_f < ymax)
				ymax = loup - abs_eps_f;
			// return ymax;
			return next_float(ymax);
		}
		else
			return loup;
	}
	bool Optimizer::update_loup(const IntervalVector &box, BoxProperties &prop)
	{
		try
		{
			pair<IntervalVector, double> p = loup_finder.find(box, loup_point, loup, prop);
			loup_point = p.first;
			loup = p.second;
			if (trace)
			{
				cout << " ";
				cout << "\033[32m loup= " << loup << "\033[0m" << endl;
				// cout << " loup point=";
				// if (loup_finder.rigorous())
				// cout << loup_point << endl;
				// else
				// cout << loup_point.lb() << endl;
			}
			return true;
		}
		catch (LoupFinder::NotFound &)
		{
			return false;
		}
	}
	// bool Optimizer::update_entailed_ctr(const IntervalVector& box) {
	// for (int j=0; j<m; j++) {
	// if (entailed->normalized(j)) {
	// continue;
	// }
	// Interval y=sys.ctrs[j].f.eval(box);
	// if (y.lb()>0) return false;
	// else if (y.ub()<=0) {
	// entailed->set_normalized_entailed(j);
	// }
	// }
	// return true;
	//}
	void Optimizer::update_uplo()
	{
		double new_uplo = POS_INFINITY;
		if (!buffer.empty())
		{
			new_uplo = buffer.minimum();
			if (new_uplo > loup && uplo_of_epsboxes > loup)
			{
				cout << " loup = " << loup << " new_uplo=" << new_uplo << " uplo_of_epsboxes=" << uplo_of_epsboxes << endl;
				ibex_error("optimizer: new_uplo>loup (please report bug)");
			}
			if (new_uplo < uplo)
			{
				cout << "uplo= " << uplo << " new_uplo=" << new_uplo << endl;
				ibex_error("optimizer: new_uplo<uplo (please report bug)");
			}
			// uplo <- max(uplo, min(new_uplo, uplo_of_epsboxes))
			if (new_uplo < uplo_of_epsboxes)
			{
				if (new_uplo > uplo)
				{
					uplo = new_uplo;
					if (trace)
						cout << "\033[33m uplo= " << uplo << "\033[0m" << endl;
				}
			}
			else
				uplo = uplo_of_epsboxes;
		}
		else if (buffer.empty() && loup != POS_INFINITY)
		{
			// empty buffer : new uplo is set to ymax (loup - precision) if a loup has been found
			new_uplo = compute_ymax(); // not new_uplo=loup, because constraint y <= ymax was enforced
			// cout << " new uplo buffer empty " << new_uplo << " uplo " << uplo << endl;
			double m = (new_uplo < uplo_of_epsboxes) ? new_uplo : uplo_of_epsboxes;
			if (uplo < m)
				uplo = m; // warning: hides the field "m" of the class
						  // note: we always have uplo <= uplo_of_epsboxes but we may have uplo > new_uplo, because
						  // ymax is strictly lower than the loup.
		}
	}
	void Optimizer::update_uplo_of_epsboxes(double ymin)
	{
		// the current box cannot be bisected. ymin is a lower bound of the objective on this box
		// uplo of epsboxes can only go down, but not under uplo : it is an upperbound for uplo,
		// that indicates a lowerbound for the objective in all the small boxes
		// found by the precision criterion
		assert(uplo_of_epsboxes >= uplo);
		assert(ymin >= uplo);
		if (uplo_of_epsboxes > ymin)
		{
			uplo_of_epsboxes = ymin;
			if (trace)
			{
				cout << " unprocessable tiny box: now uplo<=" << setprecision(12) << uplo_of_epsboxes << " uplo=" << uplo << endl;
			}
		}
	}
	void Optimizer::handle_cell(Cell &c)
	{
		contract_and_bound(c);
		if (c.box.is_empty())
		{
			delete &c;
		}
		else
		{
			buffer.push(&c);
		}
	}
	void Optimizer::contract_and_bound(Cell &c)
	{
		/*======================== contract y with y<=loup ========================*/
		Interval &y = c.box[goal_var];
		double ymax;
		if (loup == POS_INFINITY)
			ymax = POS_INFINITY;
		// ymax is slightly increased to favour subboxes of the loup
		// TODO: useful with double heap??
		else
			ymax = compute_ymax() + 1.e-15;
		y &= Interval(NEG_INFINITY, ymax);
		if (y.is_empty())
		{
			c.box.set_empty();
			return;
		}
		else
		{
			c.prop.update(BoxEvent(c.box, BoxEvent::CONTRACT, BitSet::singleton(n + 1, goal_var)));
		}
		/*================ contract x with f(x)=y and g(x)<=0 ================*/
		// cout << " [contract] x before=" << c.box << endl;
		// cout << " [contract] y before=" << y << endl;
		ContractContext context(c.prop);
		if (c.bisected_var != -1)
		{
			context.impact.clear();
			context.impact.add(c.bisected_var);
			context.impact.add(goal_var);
		}
		ctc.contract(c.box, context);
		// cout << c.prop << endl;
		if (c.box.is_empty())
			return;
		// cout << " [contract] x after=" << c.box << endl;
		// cout << " [contract] y after=" << y << endl;
		/*====================================================================*/
		/*========================= update loup =============================*/
		IntervalVector tmp_box(n);
		read_ext_box(c.box, tmp_box);
		c.prop.update(BoxEvent(c.box, BoxEvent::CHANGE));
		bool loup_ch = update_loup(tmp_box, c.prop);
		// update of the upper bound of y in case of a new loup found
		if (loup_ch)
		{
			y &= Interval(NEG_INFINITY, compute_ymax());
			c.prop.update(BoxEvent(c.box, BoxEvent::CONTRACT, BitSet::singleton(n + 1, goal_var)));
		}
		// TODO: should we propagate constraints again?
		loup_changed |= loup_ch;
		if (y.is_empty())
		{ // fix issue #44
			c.box.set_empty();
			return;
		}
		/*====================================================================*/
		// Note: there are three different cases of "epsilon" box,
		// - NoBisectableVariableException raised by the bisector (---> see optimize(...)) which
		// is independent from the optimizer
		// - the width of the box is less than the precision given to the optimizer ("eps_x" for
		// the original variables and "abs_eps_f" for the goal variable)
		// - the extended box has no bisectable domains (if eps_x=0 or <1 ulp)
		if (((tmp_box.diam() - eps_x).max() <= 0 && y.diam() <= abs_eps_f) || !c.box.is_bisectable())
		{
			update_uplo_of_epsboxes(y.lb());
			c.box.set_empty();
			return;
		}
		// ** important: ** must be done after upper-bounding
		// kkt.contract(tmp_box);
		if (tmp_box.is_empty())
		{
			c.box.set_empty();
		}
		else
		{
			// the current extended box in the cell is updated
			write_ext_box(tmp_box, c.box);
		}
	}
	Optimizer::Status Optimizer::optimize(const IntervalVector &init_box, double obj_init_bound)
	{
		/*
		// **NUEVO**: Mostrar información y volumen de la caja inicial (espacio de decisión)
		if (trace > 0 && !init_box.is_empty() && init_box.size() == this->n) {
		// Asumimos que 'this->n' es el número de variables originales del problema
		cout << "[Optimizer] Problema con " << this->n << " variables de decisión." << endl;
		cout << "[Optimizer] Caja inicial (espacio de decisión): " << init_box << endl;
		double initial_volume = init_box.volume();
		if (initial_volume == POS_INFINITY && init_box.max_diam() == POS_INFINITY) {
		cout << "[Optimizer] Volumen de caja inicial (espacio de decisión): Infinito (una o más dimensiones son no acotadas)" << endl;
		} else {
		cout << "[Optimizer] Volumen de caja inicial (espacio de decisión): " << initial_volume << endl;
		}
		}
		// --- FIN DE LA MODIFICACIÓN ---
		*/
		start(init_box, obj_init_bound);
		return optimize();
	}
	Optimizer::Status Optimizer::optimize(const CovOptimData &data, double obj_init_bound)
	{
		start(data, obj_init_bound);
		return optimize();
	}
	Optimizer::Status Optimizer::optimize(const char *cov_file, double obj_init_bound)
	{
		CovOptimData data(cov_file);
		start(data, obj_init_bound);
		return optimize();
	}
	void Optimizer::start(const IntervalVector &init_box, double obj_init_bound)
	{
		loup = obj_init_bound;
		// Just to initialize the "loup" for the buffer
		// TODO: replace with a set_loup function
		buffer.contract(loup);
		uplo = NEG_INFINITY;
		uplo_of_epsboxes = POS_INFINITY;
		nb_cells = 0;
		buffer.flush();
		Cell *root = new Cell(IntervalVector(n + 1));
		write_ext_box(init_box, root->box);
		// add data required by the bisector
		bsc.add_property(init_box, root->prop);
		// add data required by the contractor
		ctc.add_property(init_box, root->prop);
		// add data required by the buffer
		buffer.add_property(init_box, root->prop);
		// add data required by the loup finder
		loup_finder.add_property(init_box, root->prop);
		// cout << "**** Properties ****\n" << root->prop << endl;
		loup_changed = false;
		initial_loup = obj_init_bound;
		loup_point = init_box; //.set_empty();
		time = 0;
		if (cov)
			delete cov;
		cov = new CovOptimData(extended_COV ? n + 1 : n, extended_COV);
		cov->data->_optim_time = 0;
		cov->data->_optim_nb_cells = 0;
		if (trace >= 1) {
        	double initial_volume = calculate_box_volume(init_box);
        	cout << "[Optimizer START] Initial decision space volume (" << init_box.size() << " vars): " 
        	     << initial_volume << endl;
    	}

		handle_cell(*root);
	}
	void Optimizer::start(const CovOptimData &data, double obj_init_bound)
	{
		loup = obj_init_bound;
		// Just to initialize the "loup" for the buffer
		// TODO: replace with a set_loup function
		buffer.contract(loup);
		uplo = data.uplo();
		loup = data.loup();
		loup_point = data.loup_point();
		uplo_of_epsboxes = POS_INFINITY;
		nb_cells = 0;
		buffer.flush();
		for (size_t i = loup_point.is_empty() ? 0 : 1; i < data.size(); i++)
		{
			IntervalVector box(n + 1);
			if (data.is_extended_space())
				box = data[i];
			else
			{
				write_ext_box(data[i], box);
				box[goal_var] = Interval(uplo, loup);
				ctc.contract(box);
				if (box.is_empty())
					continue;
			}
			Cell *cell = new Cell(box);
			// add data required by the cell buffer
			buffer.add_property(box, cell->prop);
			// add data required by the bisector
			bsc.add_property(box, cell->prop);
			// add data required by the contractor
			ctc.add_property(box, cell->prop);
			// add data required by the loup finder
			loup_finder.add_property(box, cell->prop);
			buffer.push(cell);
		}
		loup_changed = false;
		initial_loup = obj_init_bound;
		time = 0;
		if (cov)
			delete cov;
		cov = new CovOptimData(extended_COV ? n + 1 : n, extended_COV);
		cov->data->_optim_time = data.time();
		cov->data->_optim_nb_cells = data.nb_cells();
	}
	Optimizer::Status Optimizer::optimize()
	{
		Timer timer;
		timer.start();
		update_uplo();
		try
		{
			cout << "Inicio Clustering normal" << endl;
			while (!buffer.empty())
			{
				// cout << buffer.size() << endl;
				loup_changed = false;
				// for double heap , choose randomly the buffer : top has to be called before pop
				Cell *c = buffer.top();
				if (trace >= 2)
					cout << " current box " << c->box << endl;
				try
				{
					pair<Cell *, Cell *> new_cells = bsc.bisect(*c);
					buffer.pop();
					delete c;	   // deletes the cell.
					nb_cells += 2; // counting the cells handled ( in previous versions nb_cells was the number of cells put into the buffer after being handled)
					handle_cell(*new_cells.first);
					handle_cell(*new_cells.second);
					if (uplo_of_epsboxes == NEG_INFINITY)
					{
						break;
					}
					if (loup_changed)
					{
						// In case of a new upper bound (loup_changed == true), all the boxes
						// with a lower bound greater than (loup - goal_prec) are removed and deleted.
						// Note: if contraction was before bisection, we could have the problem
						// that the current cell is removed by contractHeap. See comments in
						// older version of the code (before revision 284).
						double ymax = compute_ymax();
						buffer.contract(ymax);
						// cout << " now buffer is contracted and min=" << buffer.minimum() << endl;
						// TODO: check if happens. What is the return code in this case?
						if (ymax <= NEG_INFINITY)
						{
							if (trace)
								cout << " infinite value for the minimum " << endl;
							break;
						}
					}
					update_uplo();
					// ── Control de estancamiento (iteraciones o tamaño del buffer) ─────────
					if (loup_changed)
					{
						stagnation_counter = 0;
					}
					if (loup != POS_INFINITY)
					{
						++stagnation_counter;
					}
					// Dispara reinicio si se cumple cualquiera de los dos umbrales
					if (stagnation_counter >= restart_threshold || buffer.size() >= node_threshold)
					{
						if (trace)
						{
							cout << ">>> Reinicio clustering tras ";
							if (stagnation_counter >= restart_threshold)
								cout << stagnation_counter << " iteraciones sin mejora";
							else
								cout << "buffer.size()==" << buffer.size();
							cout << ".\n";
						}
						cluster_restart();
						stagnation_counter = 0;

						// reset counters

						restart_threshold += (restart_threshold * 2);
						node_threshold += node_threshold * 2;
						continue; // volvemos a la cabecera del while con nuevas cajas
					}
					// ───────────────────────────────────────────────────────────────────────
					if (!anticipated_upper_bounding) // useless to check precision on objective if 'true'
						if (get_obj_rel_prec() < rel_eps_f || get_obj_abs_prec() < abs_eps_f)
							break;
					if (timeout > 0)
						timer.check(timeout); // TODO: not reentrant, JN: done
					time = timer.get_time();
				}
				catch (NoBisectableVariableException &)
				{
					update_uplo_of_epsboxes((c->box)[goal_var].lb());
					buffer.pop();
					delete c;	   // deletes the cell.
					update_uplo(); // the heap has changed -> recalculate the uplo (eg: if not in best-first search)
				}
			}
			timer.stop();
			time = timer.get_time();
			// No solution found and optimization stopped with empty buffer
			// before the required precision is reached => means infeasible problem
			if (uplo_of_epsboxes == NEG_INFINITY)
				status = UNBOUNDED_OBJ;
			else if (uplo_of_epsboxes == POS_INFINITY && (loup == POS_INFINITY || (loup == initial_loup && abs_eps_f == 0 && rel_eps_f == 0)))
				status = INFEASIBLE;
			else if (loup == initial_loup)
				status = NO_FEASIBLE_FOUND;
			else if (get_obj_rel_prec() > rel_eps_f && get_obj_abs_prec() > abs_eps_f)
				status = UNREACHED_PREC;
			else
				status = SUCCESS;
		}
		catch (TimeOutException &)
		{
			status = TIME_OUT;
		}
		/* TODO: cannot retrieve variable names here. */
		for (int i = 0; i < (extended_COV ? n + 1 : n); i++)
			cov->data->_optim_var_names.push_back(string(""));
		cov->data->_optim_optimizer_status = (unsigned int)status;
		cov->data->_optim_uplo = uplo;
		cov->data->_optim_uplo_of_epsboxes = uplo_of_epsboxes;
		cov->data->_optim_loup = loup;
		cov->data->_optim_time += time;
		cov->data->_optim_nb_cells += nb_cells;
		cov->data->_optim_loup_point = loup_point;
		// for conversion between original/extended boxes
		IntervalVector tmp(extended_COV ? n + 1 : n);
		// by convention, the first box has to be the loup-point.
		if (extended_COV)
		{
			write_ext_box(loup_point, tmp);
			tmp[goal_var] = Interval(uplo, loup);
			cov->add(tmp);
		}
		else
		{
			cov->add(loup_point);
		}
		while (!buffer.empty())
		{
			Cell *cell = buffer.top();
			if (extended_COV)
				cov->add(cell->box);
			else
			{
				read_ext_box(cell->box, tmp);
				cov->add(tmp);
			}
			delete buffer.pop();
		}
		return status;
	}
	namespace
	{
		const char *green()
		{
#ifndef _WIN32
			return "\033[32m";
#else
			return "";
#endif
		}
		const char *red()
		{
#ifndef _WIN32
			return "\033[31m";
#else
			return "";
#endif
		}
		const char *white()
		{
#ifndef _WIN32
			return "\033[0m";
#else
			return "";
#endif
		}
	}
	void Optimizer::report()
	{
		if (!cov || !buffer.empty())
		{ // not started
			cout << " not started." << endl;
			return;
		}
		switch (status)
		{
		case SUCCESS:
			cout << green() << " optimization successful!" << endl;
			break;
		case INFEASIBLE:
			cout << red() << " infeasible problem" << endl;
			break;
		case NO_FEASIBLE_FOUND:
			cout << red() << " no feasible point found (the problem may be infeasible)" << endl;
			break;
		case UNBOUNDED_OBJ:
			cout << red() << " possibly unbounded objective (f*=-oo)" << endl;
			break;
		case TIME_OUT:
			cout << red() << " time limit " << timeout << "s. reached " << endl;
			break;
		case UNREACHED_PREC:
			cout << red() << " unreached precision" << endl;
			break;
		}
		cout << white() << endl;
		// No solution found and optimization stopped with empty buffer
		// before the required precision is reached => means infeasible problem
		if (status == INFEASIBLE)
		{
			cout << " infeasible problem " << endl;
		}
		else
		{
			cout << " f* in\t[" << uplo << "," << loup << "]" << endl;
			cout << "\t(best bound)" << endl
				 << endl;
			if (loup == initial_loup)
				cout << " x* =\t--\n\t(no feasible point found)" << endl;
			else
			{
				if (loup_finder.rigorous())
					cout << " x* in\t" << loup_point << endl;
				else
					cout << " x* =\t" << loup_point.lb() << endl;
				cout << "\t(best feasible point)" << endl;
			}
			cout << endl;
			double rel_prec = get_obj_rel_prec();
			double abs_prec = get_obj_abs_prec();
			cout << " relative precision on f*:\t" << rel_prec;
			if (rel_prec <= rel_eps_f)
				cout << green() << " [passed] " << white();
			cout << endl;
			cout << " absolute precision on f*:\t" << abs_prec;
			if (abs_prec <= abs_eps_f)
				cout << green() << " [passed] " << white();
			cout << endl;
		}
		cout << " cpu time used:\t\t\t" << time << "s";
		if (cov->time() != time)
			cout << " [total=" << cov->time() << "]";
		cout << endl;
		cout << " number of cells:\t\t" << nb_cells;
		if (cov->nb_cells() != nb_cells)
			cout << " [total=" << cov->nb_cells() << "]";
		cout << endl
			 << endl;
		if (statistics)
			cout << " ===== Statistics ====" << endl
				 << endl
				 << *statistics << endl;
	}



    void Optimizer::cluster_restart()
    {
        if (trace)
            cout << "[cluster_restart] Iniciando reinicio por clustering ("
                 << (clustering_params.choice == ClusteringParams::Algorithm::DBSCAN ? "DBSCAN" : (clustering_params.choice == ClusteringParams::Algorithm::KMEDOIDS ? "K-Medoids" : "K-Means"))
                 << ")...\n";

        // 1) Sacar TODAS las celdas del buffer
        std::vector<Cell *> active_cells;
        while (!buffer.empty())
            active_cells.push_back(buffer.pop());

        size_t N = active_cells.size();
        if (N == 0)
        {
            if (trace)
                cout << "[cluster_restart] buffer vacío, nada que hacer.\n";
            return;
        }

        if (trace)
            cout << "[cluster_restart] celdas extraídas: " << N << "\n";

		// calcular volumen
        double sum_of_original_volumes = 0.0;
        bool any_original_volume_is_infinite = false;
        for (Cell* cell_ptr : active_cells) {
            if (cell_ptr) { // Comprobación por si acaso
                double vol = calculate_box_volume(cell_ptr->box); // Usando tu función calculate_box_volume
                if (vol == POS_INFINITY) { // POS_INFINITY está definido en <cmath> o <limits>
                    any_original_volume_is_infinite = true;
                    // Si un volumen es infinito, la suma total será infinita
                }
                if (!any_original_volume_is_infinite) {
                    sum_of_original_volumes += vol;
                } else {
                    sum_of_original_volumes = POS_INFINITY; // Marcar la suma como infinita
                }
            }
        }

        if (trace >= 1) {
            if (any_original_volume_is_infinite) {
                cout << "[cluster_restart] Suma de volúmenes PRE-CLUSTERING (celdas originales): POS_INFINITY" << endl;
            } else {
                cout << "[cluster_restart] Suma de volúmenes PRE-CLUSTERING (celdas originales): " << sum_of_original_volumes << endl;
            }
        }

        // **** FIN DEL CÁLCULO DE VOLUMEN PRE-CLUSTERING ****

        const int dim = n + 1;

        // 2) Calcular centros
        std::vector<Point> centers;
        centers.reserve(N);
        for (Cell *c_ptr : active_cells)
        {
            const IntervalVector &box = c_ptr->box;
            Point p(dim);
            for (int j = 0; j < dim; ++j)
            {
                p[j] = box[j].mid();
            }
            centers.push_back(std::move(p));
        }

        // 3) Ejecutar Clustering 
        std::vector<int> result_labels;
        int actual_num_clusters = 0;

        // Variables para el log de VOLUMEN POST-CLUSTERING 
        double sum_of_hulls_volume_created = 0.0;
        int num_hulls_actually_formed = 0;
        bool any_formed_hull_volume_is_infinite = false;
        
        // --- SELECCIÓN Y EJECUCIÓN DEL ALGORITMO DE CLUSTERING ---
        if (clustering_params.choice == ClusteringParams::Algorithm::DBSCAN)
        {
            double eps_to_use_for_dbscan = clustering_params.eps;
            if (clustering_params.use_dynamic_eps)
            {
                if (N > 0)
                {
                    eps_to_use_for_dbscan = estimate_dynamic_eps(centers,
                                                                 clustering_params.minPts,
                                                                 clustering_params.dynamic_eps_fraction,
                                                                 clustering_params.dynamic_eps_fallback);
                    if (trace)
                        cout << "[cluster_restart] DBSCAN: Usando eps dinámico estimado = " << eps_to_use_for_dbscan << endl;
                }
                else
                {
                    if (trace)
                        cout << "[cluster_restart] DBSCAN: No hay puntos para estimar eps dinámico, usando fallback o fijo." << endl;
                }
            }
            else
            {
                if (trace)
                    cout << "[cluster_restart] DBSCAN: Usando eps fijo = " << eps_to_use_for_dbscan << endl;
            }

            if (N > 0 && N < (size_t)clustering_params.minPts)
            {
                if (trace)
                    cout << "[cluster_restart] DBSCAN: No hay suficientes puntos (" << N
                         << ") para minPts=" << clustering_params.minPts
                         << ". Marcando todos como ruido.\n"; 
                result_labels.assign(N, DBSCAN_NOISE); 
                actual_num_clusters = 0; 
            }
            else if (N == 0)
            {
                actual_num_clusters = 0;
            }
            else
            {
                DbscanClusterResult dbscan_res = dbscan(centers, eps_to_use_for_dbscan, clustering_params.minPts);
                result_labels = dbscan_res.labels;
                actual_num_clusters = dbscan_res.num_clusters;
            }

            if (trace)
            {
                cout << "[cluster_restart] DBSCAN -> " << actual_num_clusters << " clústeres encontrados (eps="
                     << eps_to_use_for_dbscan << ", minPts=" << clustering_params.minPts << ").\n";
                if (N > 0)
                {
                    int noise_count = 0;
                    for (size_t i = 0; i < N; ++i)
                        if (result_labels[i] == DBSCAN_NOISE)
                            noise_count++;
                    if (noise_count > 0)
                        cout << " DBSCAN Ruido: " << noise_count << " de " << N << " nodos" << " o sea: " << (double)noise_count / N * 100 << "%\n";
                }
            }
        }
        else if (clustering_params.choice == ClusteringParams::Algorithm::KMEDOIDS)
        {
            int k_for_kmedoids = clustering_params.k;
            if (N > 0 && (size_t)k_for_kmedoids > N)
            {
                if (trace) cout << "[cluster_restart] K-Medoids: k (" << k_for_kmedoids << ") > N (" << N << "). Usando k=N.\n";
                k_for_kmedoids = N;
            }
            
            if (N == 0) {
                actual_num_clusters = 0;
            } else if (k_for_kmedoids == 0 && N > 0) { 
                 if (trace) cout << "[cluster_restart] K-Medoids: k=0 con N>0 puntos. Agrupando todo en un clúster.\n";
                 result_labels.assign(N,0);
                 actual_num_clusters = 1;
            } else if (N > 0) { // Asegurar que k_for_kmedoids es válido si N > 0
                 ClusterResult kmedoids_res = kmedoids(centers, k_for_kmedoids, clustering_params.kmedoids_max_iters);
                 result_labels = kmedoids_res.labels;
                 actual_num_clusters = kmedoids_res.n_clusters;
            } else { // N == 0 ya cubierto, esto es por si k_for_kmedoids fuera inválido con N > 0
                 actual_num_clusters = 0; // O manejar error
            }

            if (trace)
            {
                cout << "[cluster_restart] K-Medoids -> " << actual_num_clusters
                     << " clústeres (k solicitado=" << k_for_kmedoids 
                     << ", iters=" << clustering_params.kmedoids_max_iters << ").\n";
            }
        }
        else // KMEANS (tu default)
        {
            int k_for_kmeans = (N > 0) ? std::max(1, (int)std::sqrt((double)N / 2.0)) : 0; // Evitar sqrt de negativo o cero si N=0

            if (N > 0 && (size_t)k_for_kmeans > N) {
                 if (trace) cout << "[cluster_restart] K-Means: k (" << k_for_kmeans << ") > N (" << N << "). Usando k=N.\n";
                k_for_kmeans = N;
            }

            if (N == 0) {
                actual_num_clusters = 0;
            } else if (k_for_kmeans == 0 && N > 0) { // Si k=0 por alguna razón con N>0
                if (trace) cout << "[cluster_restart] K-Means: k=0 con N>0 puntos. Agrupando todo en un clúster.\n";
                result_labels.assign(N,0);
                actual_num_clusters = 1;
            } else if (N > 0) { // Asegurar N > 0 antes de llamar a kmeans
                 ClusterResult kmeans_res = kmeans(centers, k_for_kmeans);
                 result_labels = kmeans_res.labels;
                 actual_num_clusters = kmeans_res.n_clusters;
            } else { // N == 0 ya cubierto
                 actual_num_clusters = 0;
            }
            if (trace)
            {
                cout << "[cluster_restart] K-Means -> " << actual_num_clusters
                     << " clústeres (k solicitado=" << k_for_kmeans << ").\n";
            }
        }

     // --- PROCESAMIENTO DE CLÚSTERES Y CREACIÓN DE HULLS  ---
        std::vector<Cell*> noise_cells_to_reinsert; 
        
        if (actual_num_clusters > 0 && N > 0)
        {
            std::vector<IntervalVector> hulls(actual_num_clusters, IntervalVector(dim)); 
            std::vector<bool> hull_initialized(actual_num_clusters, false); 

            for (auto &hv : hulls) 
            {
                for (int j = 0; j < dim; ++j)
                {
                    hv[j].set_empty();
                }
            }

            // Asignar celdas a hulls o a noise_cells_to_reinsert
            // y marcar las celdas originales en 'active_cells' como nullptr para indicar que fueron procesadas
            for (size_t i = 0; i < N; ++i)
            {
                if (active_cells[i] == nullptr) continue; // Ya procesada (no debería ocurrir aquí)

                int lbl = result_labels[i];

                if ((clustering_params.choice == ClusteringParams::Algorithm::DBSCAN && lbl == DBSCAN_NOISE) || lbl < 0 || lbl >= actual_num_clusters ) { 
                    noise_cells_to_reinsert.push_back(active_cells[i]);
                } else { // Pertenece a un clúster válido lbl
                    const auto &box_i = active_cells[i]->box;
                    for (int j = 0; j < dim; ++j)
                    {
                        if (!hull_initialized[lbl]) { 
                            hulls[lbl] = box_i;       
                            hull_initialized[lbl] = true;
                        } else if (!box_i[j].is_empty()) { 
                             if (hulls[lbl][j].is_empty()) { 
                                hulls[lbl][j] = box_i[j];
                            } else {
                                hulls[lbl][j] = hulls[lbl][j] | box_i[j];
                            }
                        }
                    }
                    delete active_cells[i]; // La celda original se elimina porque contribuyó a un hull
                }
                active_cells[i] = nullptr; // Marcar como procesada
            }

            // Repoblar buffer con los hulls generados
            for (int c = 0; c < actual_num_clusters; ++c)
            {
                const auto &hb_const = hulls[c];
                bool is_hull_genuinely_empty = true; 
                if (hull_initialized[c]) {
                    for(int j=0; j<dim; ++j) {
                        if (!hb_const[j].is_empty()) { is_hull_genuinely_empty = false; break; }
                    }
                }
                if (is_hull_genuinely_empty) {
                    if (trace >=1) cout << "[cluster_restart] Hull para clúster " << c << " está vacío. Descartando.\n";
                    continue;
                }

                double current_hull_volume = calculate_box_volume(hb_const);
                // num_hulls_actually_formed se incrementa aquí 
                num_hulls_actually_formed++;


                if (trace >= 2) {
                    cout << "[cluster_restart] Hull formado para clúster " << c 
                              << ". Volumen: " << current_hull_volume 
                              << ". Y: " << hb_const[goal_var] << endl;
                }

                // Actualizar suma de volúmenes de hulls
                if (current_hull_volume == POS_INFINITY) {
                    any_formed_hull_volume_is_infinite = true;
                }
                if (!any_formed_hull_volume_is_infinite) { // Solo sumar si ninguno anterior fue infinito
                    sum_of_hulls_volume_created += current_hull_volume;
                } else { // Si alguno ya fue infinito, la suma total es infinita
                    sum_of_hulls_volume_created = POS_INFINITY;
                }


                Cell *nc = new Cell(hb_const); 
                buffer.add_property(nc->box, nc->prop); 
                bsc.add_property(nc->box, nc->prop);
                ctc.add_property(nc->box, nc->prop);
                loup_finder.add_property(nc->box, nc->prop);
                buffer.push(nc);
                if (trace >=1) cout << "[cluster_restart] Push hull para clúster " << c << " [" << nc->box << "]\n";
            }
        }
        else // No se formaron clústeres (actual_num_clusters == 0) pero N > 0
        {
            if (trace >=1 && N > 0) {
                 cout << "[cluster_restart] No se formaron clústeres. Reinsertando " << N << " celdas originales.\n";
            }
            // Mover todas las celdas de active_cells a noise_cells_to_reinsert
            for (Cell *cell_ptr : active_cells) {
                if (cell_ptr) noise_cells_to_reinsert.push_back(cell_ptr); 
            }
        }
        // Limpiar active_cells, ya que todos sus punteros válidos deberían estar ahora
        // en noise_cells_to_reinsert o haber sido eliminados.
        active_cells.clear(); 

        // Reinsertar celdas de ruido (o todas las originales si no hubo clústeres)
        if (!noise_cells_to_reinsert.empty()) {
            if (trace >=1) cout << "[cluster_restart] Reinsertando " << noise_cells_to_reinsert.size() << " celdas (ruido/no agrupadas) al buffer.\n";
            for (Cell* noise_cell : noise_cells_to_reinsert) {
                if (noise_cell) buffer.push(noise_cell); 
            }
        }
        noise_cells_to_reinsert.clear();

        // ---- Log de la SUMA de volúmenes de hulls POST-CLUSTERING ----
        if (trace >= 1) {
            if (num_hulls_actually_formed > 0) {
                if (any_formed_hull_volume_is_infinite) {
                    cout << "[cluster_restart] Suma de volúmenes POST-CLUSTERING (" << num_hulls_actually_formed 
                              << " hulls formados): POS_INFINITY." << endl;
                } else {
                    cout << "[cluster_restart] Suma de volúmenes POST-CLUSTERING (" << num_hulls_actually_formed 
                              << " hulls formados): " << sum_of_hulls_volume_created << endl;
                }
            } else if (N > 0) { 
                cout << "[cluster_restart] No se formaron hulls POST-CLUSTERING (0 clústeres o todos vacíos)." << endl;
            }
        }
        // ---- Fin log suma de volúmenes ----

        if (trace >=1 )
        {
            cout << "[cluster_restart] Completado. Buffer ahora tiene " << buffer.size() << " celdas.\n";
        }
    } // Fin de Optimizer::cluster_restart
    
} // end namespace ibex