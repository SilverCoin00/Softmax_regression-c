#include "D:\Data\code_doc\AI_model_building\Softmax_regression\Core.h"

typedef struct Softmax_regression {
    Dataset_2* data;
    Weights* weights;
} Softmax_regression;

void predict(Dataset_2* data, Weights* w, float** y_pred) {    // yp ij = exp(z ij) / sum j(exp(z ij))
    if (!y_pred) return ;
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
        for (j = 0; j < w->classes; j++) y_pred[i][j] = z[i][j] / sum_z;
    }
    free_matrix(z, data->samples);
}
float loss_func(float** y_pred, float** y_true, int classes, int samples) {
    float loss = 0;
    float e = 1e-10;
    for (int i = 0, j; i < samples; i++) {
        for (j = 0; j < classes; j++) {
            if (y_true[i][j] == 1.0) loss += logf(y_pred[i][j] + e);
        }
    }
    return -loss / samples;
}
void train(Softmax_regression* model, char* GD_type, int iteration, float learning_rate, int batch_size) {
    float** y_pred = new_matrix(model->data->samples, model->weights->classes);
    float loss;
    float** pre_velo = new_matrix(model->weights->num_weights, model->weights->classes);
    int* random_i = (int*)malloc(model->data->samples* sizeof(int)), i, loop;
    for (i = 0; i < model->data->samples; i++) random_i[i] = i;
	Dataset_2* batch;

    if (batch_size <= 0 || batch_size >= model->data->samples) batch_size = model->data->samples;
	else shuffle_index(random_i, model->data->samples, i);
	loop = model->data->samples / batch_size;

    while (iteration > 0) {
		predict(model->data, model->weights, y_pred);
        shuffle_index(random_i, model->data->samples, i);
        
		for (i = 0; i < loop; i++) {
			batch = dataset2_samples_order_copy(model->data, random_i, i* batch_size, (i + 1)* batch_size);
            
			if (!strcmp(GD_type, "GD")) grad_descent(batch, model->weights, learning_rate);
			else if (!strcmp(GD_type, "GDM")) grad_descent_momentum(batch, model->weights, learning_rate, pre_velo, 0.9);
			else if (!strcmp(GD_type, "NAG")) nesterov_accelerated_grad(batch, model->weights, learning_rate, pre_velo, 0.9);
			free_dataset2(batch);
		}
		if (batch_size* loop < model->data->samples) {
			batch = dataset2_samples_order_copy(model->data, random_i, i* batch_size, model->data->samples);

			if (!strcmp(GD_type, "GD")) grad_descent(batch, model->weights, learning_rate);
			else if (!strcmp(GD_type, "GDM")) grad_descent_momentum(batch, model->weights, learning_rate, pre_velo, 0.9);
			else if (!strcmp(GD_type, "NAG")) nesterov_accelerated_grad(batch, model->weights, learning_rate, pre_velo, 0.9);
			free_dataset2(batch);
		}
		loss = loss_func(y_pred, model->data->y, model->data->y_types, model->data->samples);
		printf("Iteration left: %d, loss = %.6f\n", iteration, loss);
		print_weights(model->weights, 8);
        printf("\n");
		iteration--;
	}
	free(random_i);
	free_matrix(pre_velo, model->weights->num_weights);
	free_matrix(y_pred, model->data->samples);
}
void free_sm_model(Softmax_regression* model) {
    if (model->data) free_dataset2(model->data);
    if (model->weights) free_weights(model->weights);
    free(model);
}