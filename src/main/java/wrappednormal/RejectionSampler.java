package wrappednormal;

import java.util.Random;

/**
 * Rejection sampler using the tightest component as proposal.
 */
public class RejectionSampler implements ProductSampler {
    private final WrappedBivariateNormal[] components;
    private final int numComponents;
    private Random random;
    private final SamplerStatistics stats;

    private int tightestComponent;
    private double logMaxDensityRatio;

    public RejectionSampler(WrappedBivariateNormal[] components) {
        this.components = components;
        this.numComponents = components.length;
        this.random = new Random();
        this.stats = new SamplerStatistics();
        initialize();
    }

    @Override
    public String getName() {
        return "Rejection Sampler (Tightest Component)";
    }

    @Override
    public String getDescription() {
        return "Uses the component with smallest variance as proposal distribution. " +
                "Can be inefficient when components have little overlap.";
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
        // Find tightest component
        tightestComponent = 0;
        double minVariance = Double.POSITIVE_INFINITY;

        for (int i = 0; i < numComponents; i++) {
            double variance = components[i].getSigma1() * components[i].getSigma1() +
                    components[i].getSigma2() * components[i].getSigma2();
            if (variance < minVariance) {
                minVariance = variance;
                tightestComponent = i;
            }
        }

        // Estimate the maximum density ratio
        double[] mode = findMode(50);
        double logProductAtMode = logUnnormalizedPdf(mode[0], mode[1]);
        double logProposalAtMode = components[tightestComponent].logPdf(mode[0], mode[1]);
        logMaxDensityRatio = logProductAtMode - logProposalAtMode;
    }

    @Override
    public double[] sample() {
        long startTime = System.nanoTime();

        while (true) {
            stats.recordAttempt();

            // Sample from proposal
            double[] proposal = components[tightestComponent].sample();

            // Compute acceptance probability
            double logProduct = logUnnormalizedPdf(proposal[0], proposal[1]);
            double logProposal = components[tightestComponent].logPdf(proposal[0], proposal[1]);
            double logAcceptProb = logProduct - logProposal - logMaxDensityRatio;

            if (Math.log(random.nextDouble()) <= logAcceptProb) {
                long timeTaken = (System.nanoTime() - startTime) / 1_000_000;
                stats.recordSample(timeTaken);
                return proposal;
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
        // Update batch timing
        stats.recordBatch(0, totalTime - n * stats.getTotalTime() / Math.max(1, stats.getTotalSamples()), 0);

        return samples;
    }

    private double[] findMode(int gridSize) {
        double maxLogDensity = Double.NEGATIVE_INFINITY;
        double[] mode = new double[2];
        double delta = 2 * Math.PI / gridSize;

        for (int i = 0; i < gridSize; i++) {
            for (int j = 0; j < gridSize; j++) {
                double theta1 = i * delta;
                double theta2 = j * delta;
                double logDensity = logUnnormalizedPdf(theta1, theta2);

                if (logDensity > maxLogDensity) {
                    maxLogDensity = logDensity;
                    mode[0] = theta1;
                    mode[1] = theta2;
                }
            }
        }

        return mode;
    }

    private double logUnnormalizedPdf(double theta1, double theta2) {
        double logSum = 0;
        for (WrappedBivariateNormal comp : components) {
            logSum += comp.logPdf(theta1, theta2);
        }
        return logSum;
    }
}