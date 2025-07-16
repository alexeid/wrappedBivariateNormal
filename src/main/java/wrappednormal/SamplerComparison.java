package wrappednormal;

import java.util.ArrayList;
import java.util.List;

/**
 * Compares different sampling methods for product of wrapped bivariate normals.
 */
public class SamplerComparison {

    public static void main(String[] args) {
        System.out.println("=== Sampler Comparison for Product of Wrapped Bivariate Normals ===\n");

        // Create test distributions with varying overlap
        System.out.println("Test Case 1: High Overlap (components close together)");
        WrappedBivariateNormal[] highOverlap = {
                new WrappedBivariateNormal(Math.PI, Math.PI, 0.5, 0.5, 0.1),
                new WrappedBivariateNormal(Math.PI + 0.3, Math.PI - 0.3, 0.4, 0.4, 0.2),
                new WrappedBivariateNormal(Math.PI - 0.3, Math.PI + 0.3, 0.6, 0.6, 0.0)
        };
        runComparison(highOverlap, 10000);

        System.out.println("\nTest Case 2: Medium Overlap");
        WrappedBivariateNormal[] mediumOverlap = {
                new WrappedBivariateNormal(Math.PI/2, Math.PI, 0.3, 0.3, 0.1),
                new WrappedBivariateNormal(Math.PI, Math.PI/2, 0.5, 0.5, 0.2),
                new WrappedBivariateNormal(3*Math.PI/4, 3*Math.PI/4, 0.4, 0.4, 0.0)
        };
        runComparison(mediumOverlap, 10000);

        System.out.println("\nTest Case 3: Low Overlap (components far apart)");
        WrappedBivariateNormal[] lowOverlap = {
                new WrappedBivariateNormal(Math.PI/6, 3*Math.PI/2, 0.3, 0.3, 0.1),
                new WrappedBivariateNormal(3*Math.PI/4, Math.PI/6, 0.4, 0.4, 0.2),
                new WrappedBivariateNormal(3*Math.PI/2, 3*Math.PI/4, 0.6, 0.6, 0.0)
        };
        runComparison(lowOverlap, 10000);

        System.out.println("\nTest Case 4: Very Tight Components");
        WrappedBivariateNormal[] veryTight = {
                new WrappedBivariateNormal(1.0, 1.0, 0.35, 0.35, 0.0),
                new WrappedBivariateNormal(2.0, 2.0, 0.35, 0.35, 0.0),
                new WrappedBivariateNormal(3.0, 3.0, 0.35, 0.35, 0.0)
        };
        runComparison(veryTight, 10000);
    }

    private static void runComparison(WrappedBivariateNormal[] components, int numSamples) {
        // Create samplers
        List<ProductSampler> samplers = new ArrayList<>();
        samplers.add(new GibbsSampler(components));
        samplers.add(new MixtureImportanceSampler(components));
        samplers.add(new MetropolisGibbsSampler(components));

        // Test each sampler
        System.out.println("Generating " + numSamples + " samples with each method:");
        System.out.println("-".repeat(80));

        for (ProductSampler sampler : samplers) {
            System.out.println("\n" + sampler.getName());
            System.out.println(sampler.getDescription());

            // Reset and set seed for reproducibility
            sampler.reset();
            sampler.setSeed(12345);

            // Time the sampling
            long startTime = System.currentTimeMillis();
            double[][] samples = sampler.sample(numSamples);
            long endTime = System.currentTimeMillis();

            // Get statistics
            SamplerStatistics stats = sampler.getStatistics();

            // Compute sample statistics
            double[] means = computeSampleMeans(samples);
            double[] variances = computeSampleVariances(samples, means);

            // Print results
            System.out.printf("Time: %d ms\n", endTime - startTime);
            System.out.printf("Statistics: %s\n", stats);
            System.out.printf("Sample means: (%.4f, %.4f)\n", means[0], means[1]);
            System.out.printf("Sample variances: (%.4f, %.4f)\n", variances[0], variances[1]);

            // Check for NaN or invalid samples
            int invalidCount = 0;
            for (double[] sample : samples) {
                if (Double.isNaN(sample[0]) || Double.isNaN(sample[1]) ||
                        sample[0] < 0 || sample[0] >= 2*Math.PI ||
                        sample[1] < 0 || sample[1] >= 2*Math.PI) {
                    invalidCount++;
                }
            }
            if (invalidCount > 0) {
                System.out.println("WARNING: " + invalidCount + " invalid samples!");
            }
        }
        System.out.println("-".repeat(80));
    }

    private static double[] computeSampleMeans(double[][] samples) {
        double sumCos1 = 0, sumSin1 = 0;
        double sumCos2 = 0, sumSin2 = 0;

        for (double[] sample : samples) {
            sumCos1 += Math.cos(sample[0]);
            sumSin1 += Math.sin(sample[0]);
            sumCos2 += Math.cos(sample[1]);
            sumSin2 += Math.sin(sample[1]);
        }

        int n = samples.length;
        double mean1 = Math.atan2(sumSin1 / n, sumCos1 / n);
        double mean2 = Math.atan2(sumSin2 / n, sumCos2 / n);

        // Wrap to [0, 2π)
        if (mean1 < 0) mean1 += 2 * Math.PI;
        if (mean2 < 0) mean2 += 2 * Math.PI;

        return new double[] {mean1, mean2};
    }

    private static double[] computeSampleVariances(double[][] samples, double[] means) {
        double sumSq1 = 0, sumSq2 = 0;

        for (double[] sample : samples) {
            double diff1 = angularDistance(sample[0], means[0]);
            double diff2 = angularDistance(sample[1], means[1]);
            sumSq1 += diff1 * diff1;
            sumSq2 += diff2 * diff2;
        }

        int n = samples.length;
        return new double[] {sumSq1 / n, sumSq2 / n};
    }

    private static double angularDistance(double a, double b) {
        double diff = a - b;
        while (diff > Math.PI) diff -= 2 * Math.PI;
        while (diff < -Math.PI) diff += 2 * Math.PI;
        return diff;
    }
}