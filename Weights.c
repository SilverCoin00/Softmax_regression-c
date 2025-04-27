#include "D:\Data\code_doc\AI_model_building\Softmax_regression\Core.h"

typedef struct Weights {
    float** weights;
	int num_weights;    // x_features
	int classes;        // y_classes
} Weights;

Weights* init_weights(int num_of_features, int num_of_classes, int random_init) {
	Weights* newb = (Weights*)malloc(sizeof(Weights));
	newb->num_weights = num_of_features + 1;
	newb->classes = num_of_classes;
	newb->weights = (float**)malloc((num_of_features + 1)* sizeof(float*));
	srand(random_init);
	for (int i = 0, j; i < newb->num_weights; i++) {
		newb->weights[i] = (float*)calloc(num_of_classes, sizeof(float));
		for (j = 0; j < num_of_classes; j++) newb->weights[i][j] = ((float)rand() / RAND_MAX)* 0.2 - 0.1;
	}
	return newb;
}
float** weights_derivative(Dataset_2* data, Weights* w) {    // deriv(w) = (1 / m). X(T).(yp - yt)
	int i, j;
	float** z = matrix_multiply(data->x, w->weights, data->samples, data->features + 1, w->classes);  // X(T).W = (W(T).X)(T)
    float sum_z, max_z;
	for (i = 0; i < data->samples; i++) {
        sum_z = 0;
		max_z = z[i][0];
        for (j = 1; j < w->classes; j++) if (z[i][j] > max_z) max_z = z[i][j];
        for (j = 0; j < w->classes; j++) {
            z[i][j] = exp(z[i][j] - max_z);
            sum_z += z[i][j];
        }
        for (j = 0; j < w->classes; j++) {
			z[i][j] /= sum_z;
			z[i][j] -= data->y[i][j];
		}
    }
	float** x_T = transpose_matrix(data->x, data->samples, data->features + 1);
	float** deriv = matrix_multiply(x_T, z, data->features + 1, data->samples, w->classes);
    free_matrix(z, data->samples);
	free_matrix(x_T, data->features + 1);
	for (i = 0; i < data->features + 1; i++) {
		for (j = 0; j < w->classes; j++) deriv[i][j] /= data->samples;
	}
	return deriv;
}
void print_weights(Weights* w, int decimal) {
	printf("Weights: [");
	for (int i = 0, j; i < w->num_weights; i++) {
		if (i != 0) printf("\n%11s", "[");
		else printf("[");
		for (j = 0; j < w->classes - 1; j++) {
			printf("%.*f, ", decimal, w->weights[i][j]);
		}
		printf("%.*f]", decimal, w->weights[i][w->classes - 1]);
	}
	printf("]\n");
}
void free_weights(Weights* w) {
	for (int i = 0; i < w->num_weights; i++) free(w->weights[i]);
	free(w->weights);
	free(w);
}
void grad_descent(Dataset_2* data, Weights* w, float learning_rate) {
	float** gradient = weights_derivative(data, w);
	for (int i = 0, j; i < w->num_weights; i++) {
		for (j = 0; j < w->classes; j++) w->weights[i][j] -= learning_rate* gradient[i][j];
	}
	free_matrix(gradient, w->num_weights);
}
void grad_descent_momentum(Dataset_2* data, Weights* w, float learning_rate, float** pre_velocity, float velocity_rate) {
	float** velo = weights_derivative(data, w);
	for (int i = 0, j; i < w->num_weights; i++) {
		for (j = 0; j < w->classes; j++) {
			velo[i][j] *= learning_rate;
			velo[i][j] += velocity_rate* pre_velocity[i][j];
			w->weights[i][j] -= velo[i][j];
			pre_velocity[i][j] = velo[i][j];
		}
	}
	free_matrix(velo, w->num_weights);
}
void nesterov_accelerated_grad(Dataset_2* data, Weights* w, float learning_rate, float** pre_velocity, float velocity_rate) {
	Weights* fore_w = init_weights(w->num_weights - 1, w->classes, 1);
	int i, j;
	for (i = 0; i < w->num_weights; i++) {
		for (j = 0; j < w->classes; j++) fore_w->weights[i][j] = w->weights[i][j] - velocity_rate* pre_velocity[i][j];
	}
	float** velo = weights_derivative(data, fore_w);
	free_weights(fore_w);
	for (i = 0; i < w->num_weights; i++) {
		for (j = 0; j < w->classes; j++) {
			velo[i][j] *= learning_rate;
			velo[i][j] += velocity_rate* pre_velocity[i][j];
			w->weights[i][j] -= velo[i][j];
			pre_velocity[i][j] = velo[i][j];
		}
	}
	free_matrix(velo, w->num_weights);
}