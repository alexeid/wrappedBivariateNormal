package wrappednormal;

import java.util.Random;

/**
 * Importance sampler using a mixture of all components as proposal.
 */
public class MixtureImportanceSampler implements ProductSampler {
    private final WrappedBivariateNormal[] components;
    private final int numComponents;
    private Random random;
    private final SamplerStatistics stats;

    private double[] proposalWeights;
    private double logMaxWeight;

    public MixtureImportanceSampler(WrappedBivariateNormal[] components) {
        this.components = components;
        this.numComponents = components.length;
        this.random = new Random();
        this.stats = new SamplerStatistics();
        initialize();
    }

    @Override
    public String getName() {
        return "Mixture Importance Sampler";
    }

    @Override
    public String getDescription() {
        return "Uses a weighted mixture of all components as proposal. " +
                "Better than single-component rejection when components have different modes.";
    }

    @Override
    public void setSeed(long seed) {
        this.random = new Random(seed);
    }

    @Override
    public void reset() {
        stats.reset();
        initialize();
    }

    @Override
    public SamplerStatistics getStatistics() {
        return stats;
    }

    private void initialize() {
        // Start with equal weights
        proposalWeights = new double[numComponents];
        for (int i = 0; i < numComponents; i++) {
            proposalWeights[i] = 1.0 / numComponents;
        }

        // Estimate maximum importance weight
        logMaxWeight = Double.NEGATIVE_INFINITY;

        // Sample from each component to find high-weight regions
        for (int i = 0; i < numComponents; i++) {
            for (int j = 0; j < 100; j++) {
                double[] sample = components[i].sample();

                // Compute log weight
                double logTarget = logUnnormalizedPdf(sample[0], sample[1]);
                double logProposal = logMixturePdf(sample[0], sample[1]);
                double logWeight = logTarget - logProposal;

                if (logWeight > logMaxWeight) {
                    logMaxWeight = logWeight;
                }
            }
        }

        // Add buffer
        logMaxWeight += 0.1;
    }

    @Override
    public double[] sample() {
        long startTime = System.nanoTime();

        while (true) {
            stats.recordAttempt();

            // Choose component
            double u = random.nextDouble();
            int k = 0;
            double cumSum = 0;
            for (int i = 0; i < numComponents; i++) {
                cumSum += proposalWeights[i];
                if (u <= cumSum) {
                    k = i;
                    break;
                }
            }

            // Sample from chosen component
            double[] sample = components[k].sample();

            // Compute acceptance probability
            double logTarget = logUnnormalizedPdf(sample[0], sample[1]);
            double logProposal = logMixturePdf(sample[0], sample[1]);
            double logWeight = logTarget - logProposal;
            double logAcceptProb = logWeight - logMaxWeight;

            if (Math.log(random.nextDouble()) <= logAcceptProb) {
                long timeTaken = (System.nanoTime() - startTime) / 1_000_000;
                stats.recordSample(timeTaken);
                return sample;
            }
        }
    }

    @Override
    public double[][] sample(int n) {
        double[][] samples = new double[n][2];
        long startTime = System.currentTimeMillis();

        for (int i = 0; i < n; i++) {
            samples[i] = sample();
        }

        long totalTime = System.currentTimeMillis() - startTime;
        stats.recordBatch(0, totalTime - n * stats.getTotalTime() / Math.max(1, stats.getTotalSamples()), 0);

        return samples;
    }

    private double logUnnormalizedPdf(double theta1, double theta2) {
        double logSum = 0;
        for (WrappedBivariateNormal comp : components) {
            logSum += comp.logPdf(theta1, theta2);
        }
        return logSum;
    }

    private double logMixturePdf(double theta1, double theta2) {
        double[] logTerms = new double[numComponents];
        double maxLogTerm = Double.NEGATIVE_INFINITY;

        for (int i = 0; i < numComponents; i++) {
            logTerms[i] = Math.log(proposalWeights[i]) + components[i].logPdf(theta1, theta2);
            maxLogTerm = Math.max(maxLogTerm, logTerms[i]);
        }

        // Log-sum-exp trick
        double sumExp = 0;
        for (int i = 0; i < numComponents; i++) {
            sumExp += Math.exp(logTerms[i] - maxLogTerm);
        }

        return maxLogTerm + Math.log(sumExp);
    }
}