package com.sewerynkamil;

import weka.clusterers.ClusterEvaluation;
import weka.clusterers.SimpleKMeans;
import weka.core.Instances;
import weka.core.converters.ConverterUtils;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;
import weka.gui.explorer.ClustererAssignmentsPlotInstances;
import weka.gui.explorer.ExplorerDefaults;
import weka.gui.visualize.PlotData2D;
import weka.gui.visualize.VisualizePanel;

import javax.swing.*;
import java.awt.*;

public class KMeansWeka {
    public static void main(String[] args) throws Exception {
        ConverterUtils.DataSource source = new ConverterUtils.DataSource("data/clustering_data.csv");

        Instances dataset = source.getDataSet();
        dataset.setClassIndex(0);

        // Use a "remove attribute" filter to create training set (remove label column).
        Remove removeFilter = new Remove();
        removeFilter.setAttributeIndices("first");
        removeFilter.setInputFormat(dataset);
        Instances train = Filter.useFilter(dataset, removeFilter);

        // Get the number of classes.
        int numClasses = dataset.classAttribute().numValues();
        System.out.println(numClasses);

        // Create a k-means instance.
        SimpleKMeans kMeans = new SimpleKMeans();
        kMeans.setNumClusters(numClasses);
        kMeans.buildClusterer(train);

        // Use weka's cluster evaluation capabilities.
        ClusterEvaluation evaluation = new ClusterEvaluation();
        evaluation.setClusterer(kMeans);
        evaluation.evaluateClusterer(dataset);
        System.out.println(evaluation.clusterResultsToString());

        // Visualize the results.
        ClustererAssignmentsPlotInstances plotInstances = ExplorerDefaults.getClustererAssignmentsPlotInstances();
        plotInstances.setClusterer(kMeans);
        plotInstances.setInstances(dataset);
        plotInstances.setClusterEvaluation(evaluation);
        plotInstances.setUp();
        PlotData2D plotData = plotInstances.getPlotData("Cluster visualization");

        VisualizePanel vp = new VisualizePanel();
        vp.setName(plotData.getPlotName());
        vp.addPlot(plotData);
        vp.setXIndex(2);
        vp.setYIndex(3);

        JFrame jf = new JFrame("Weka KMeans visualizer");
        jf.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        jf.setSize(500, 500);
        jf.getContentPane().setLayout(new BorderLayout());
        jf.getContentPane().add(vp, BorderLayout.CENTER);
        jf.setVisible(true);
    }
}