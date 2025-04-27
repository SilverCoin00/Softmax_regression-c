#pragma once
#include "Pandas_&_Numpy.c"
#include "Sklearn.c"

typedef struct Dataset_2 {
	float** x;
	float** y;
    int y_types;
	int features;      // x_cols
	int samples;       // rows
} Dataset_2;

Dataset_2* trans_dframe_to_dset2(Data_Frame* df, const char* predict_feature_col) {
    float** enc_sdata = NULL;
	int y_col = strtoi(predict_feature_col), i, j, k, is_y_str = 0;
	if (df->str_cols[0] != 0) {
		enc_sdata = (float**)malloc(df->str_cols[0]* sizeof(float*));
		Label_encoder* encoder = (Label_encoder*)malloc(sizeof(Label_encoder));
		for (i = 0; i < df->str_cols[0]; i++) {
			encoder_fit(df->str_data, df->row, i, encoder, "Label_encoder");
			enc_sdata[i] = (float*) encoder_transform(df->str_data, df->row, i, encoder, "Label_encoder");
			free_set(encoder->sample_types);
		}
		free(encoder);
	}
	if (y_col < 0) {
		j = df->col + df->str_cols[0];
		for (i = 0; i < j; i++) {
			if (!strcmp(df->features[i], predict_feature_col)) {
				for (k = 1; k <= df->str_cols[0]; k++)
					if (df->str_cols[k] == i) {
						y_col = k - 1;
						is_y_str = 1;
						goto out;
					}
				y_col = i;
				break;
			}
		}
	}
	out:;
	Dataset_2* newd = (Dataset_2*)malloc(sizeof(Dataset_2));
	if (is_y_str) newd->y = label_to_one_hot_encode(NULL, 0, enc_sdata[y_col], df->row, &(newd->y_types));
	else {
		if (y_col >= 0) newd->y = label_to_one_hot_encode(df->data, y_col, NULL, df->row, &(newd->y_types));
	}
	newd->x = (float**)malloc(df->row* sizeof(float*));
	for (i = 0; i < df->row; i++) {
		newd->x[i] = (float*)malloc((df->col + df->str_cols[0])* sizeof(float));  // drop 1 col for y but plus 1 for bias, so, nothing changes
		for (j = 0, k = 0; j < df->col; k++) {
			if (!is_y_str) if (k == y_col) continue;
			newd->x[i][j++] = df->data[i][k];
		}
		for (k = 0; k < df->str_cols[0]; k++) {
			if (is_y_str) if (k == y_col) continue;
			newd->x[i][j++] = enc_sdata[k][i];
		}
		newd->x[i][df->col + df->str_cols[0] - 1] = 1;
	}
	newd->features = df->col + df->str_cols[0] - 1;
	newd->samples = df->row;
	if (enc_sdata) free_matrix(enc_sdata, df->str_cols[0]);
	return newd;
}
void dataset2_sample_copy(const Dataset_2* ds, int ds_sample_index, Dataset_2* copy, int copy_sample_index) {
	int i;
	for (i = 0; i < ds->features && i < copy->features; i++)
		copy->x[copy_sample_index][i] = ds->x[ds_sample_index][i];
	copy->x[copy_sample_index][copy->features] = 1.0;
	for (i = 0; i < ds->y_types && i < copy->y_types; i++)
		copy->y[copy_sample_index][i] = ds->y[ds_sample_index][i];
}
Dataset_2* dataset2_samples_order_copy(const Dataset_2* ds, int* order, int order_begin_index, int order_end_index) {
	Dataset_2* newd = (Dataset_2*)malloc(sizeof(Dataset_2));
	newd->x = (float**)malloc((order_end_index - order_begin_index)* sizeof(float*));
	newd->y = (float**)malloc((order_end_index - order_begin_index)* sizeof(float*));
	newd->features = ds->features;
	newd->y_types = ds->y_types;
	newd->samples = order_end_index - order_begin_index;
	for (int i = order_begin_index; i < order_end_index; i++) {
		newd->x[i - order_begin_index] = (float*)malloc((ds->features + 1)* sizeof(float));
		newd->y[i - order_begin_index] = (float*)malloc((ds->y_types)* sizeof(float));
		dataset2_sample_copy(ds, order[i], newd, i - order_begin_index);
	}
	return newd;
}
void print_dataset2(Dataset_2* ds, int decimal, int col_space, int num_of_rows) {
	if (!ds) return ;
	if (num_of_rows < 0 || num_of_rows > ds->samples) num_of_rows = ds->samples;
	printf(" Row\n");
	for (int i = 0, j; i < num_of_rows; i++) {
		printf("%4d\t", i + 1);
		for (j = 0; j < ds->features; j++) {
			printf("%*.*f ", col_space, decimal, ds->x[i][j]);
		}
        printf("\t|   [ ");
        for (j = 0; j < ds->y_types; j++) {
		    printf("%.0f ", ds->y[i][j]);
        }
        printf("]\n");
	}
}
void free_dataset2(Dataset_2* ds) {
	if (!ds) return ;
	for (int i = 0; i < ds->samples; i++) {
        free(ds->x[i]);
        free(ds->y[i]);
    }
	free(ds->x);
	free(ds->y);
	free(ds);
}
