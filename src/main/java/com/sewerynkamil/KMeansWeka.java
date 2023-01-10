package com.sewerynkamil;

import weka.core.Instances;
import weka.core.converters.ConverterUtils;

public class KMeansWeka {
    public static void main(String[] args) throws Exception {
        ConverterUtils.DataSource source = new ConverterUtils.DataSource("data/clustering_data.csv");

        Instances dataset = source.getDataSet();
        dataset.setClassIndex(0);

        System.out.println(dataset.size());
    }
}