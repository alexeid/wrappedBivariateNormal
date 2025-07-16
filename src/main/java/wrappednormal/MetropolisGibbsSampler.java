package wrappednormal;

import java.util.Random;

/**
 * Fast Metropolis-within-Gibbs sampler for product of wrapped bivariate normals.
 * Uses adaptive proposal scales and minimal evaluations per update.
 */
public class MetropolisGibbsSampler implements ProductSampler {
    private final WrappedBivariateNormal[] components;
    private final int numComponents;
    private Random random;
    private final SamplerStatistics stats;

    // Sampler state
    private boolean initialized = false;
    private double[] currentSample = new double[2];
    private double currentLogDensity;

    // Adaptive parameters
    private double scale1 = 0.5;
    private double scale2 = 0.5;
    private final int burnIn = 100;
    private final int gibbsSteps = 2; // Only 2 steps needed with good mixing

    // Adaptation tracking
    private int[] acceptances = new int[2];
    private int[] attempts = new int[2];
    private int samplesSinceAdapt = 0;
    private static final int ADAPT_INTERVAL = 100;
    private static final double TARGET_ACCEPTANCE = 0.44; // Optimal for 1D

    public MetropolisGibbsSampler(WrappedBivariateNormal[] components) {
        this.components = components;
        this.numComponents = components.length;
        this.random = new Random();
        this.stats = new SamplerStatistics();

        // Initialize scales based on component characteristics
        initializeScales();
    }

    private void initializeScales() {
        double sumSigma1 = 0, sumSigma2 = 0;
        for (WrappedBivariateNormal comp : components) {
            sumSigma1 += comp.getSigma1();
            sumSigma2 += comp.getSigma2();
        }

        // Start with average component width
        scale1 = sumSigma1 / numComponents;
        scale2 = sumSigma2 / numComponents;

        // Adjust for multiple components
        if (numComponents > 1) {
            scale1 *= 0.7; // Slightly smaller for products
            scale2 *= 0.7;
        }
    }

    @Override
    public String getName() {
        return "Metropolis-within-Gibbs Sampler";
    }

    @Override
    public String getDescription() {
        return "Fast sampler using Metropolis updates within Gibbs. " +
                "Minimal evaluations per update with adaptive scaling.";
    }

    @Override
    public void setSeed(long seed) {
        this.random = new Random(seed);
    }

    @Override
    public void reset() {
        initialized = false;
        stats.reset();
        acceptances[0] = acceptances[1] = 0;
        attempts[0] = attempts[1] = 0;
        samplesSinceAdapt = 0;
    }

    @Override
    public SamplerStatistics getStatistics() {
        return stats;
    }

    @Override
    public double[] sample() {
        long startTime = System.nanoTime();

        if (!initialized) {
            initialize();
        }

        // Gibbs steps with Metropolis updates
        for (int step = 0; step < gibbsSteps; step++) {
            updateTheta1();
            updateTheta2();
        }

        // Adapt scales periodically
        samplesSinceAdapt++;
        if (samplesSinceAdapt >= ADAPT_INTERVAL) {
            adaptScales();
            samplesSinceAdapt = 0;
        }

        long timeTaken = (System.nanoTime() - startTime) / 1_000_000;
        stats.recordSample(timeTaken);

        return new double[] {currentSample[0], currentSample[1]};
    }

    @Override
    public double[][] sample(int n) {
        double[][] samples = new double[n][2];
        long startTime = System.currentTimeMillis();

        for (int i = 0; i < n; i++) {
            samples[i] = sample();
        }

        long totalTime = System.currentTimeMillis() - startTime;
        stats.recordBatch(0, totalTime, 0);

        return samples;
    }

    private void initialize() {
        // Start from a random component's mean with small perturbation
        int idx = random.nextInt(numComponents);
        currentSample[0] = wrapAngle(components[idx].getMu1() + 0.1 * random.nextGaussian());
        currentSample[1] = wrapAngle(components[idx].getMu2() + 0.1 * random.nextGaussian());

        // Compute initial log density
        currentLogDensity = computeLogDensity(currentSample[0], currentSample[1]);

        // Burn-in
        for (int i = 0; i < burnIn; i++) {
            updateTheta1();
            updateTheta2();
        }

        initialized = true;
    }

    private void updateTheta1() {
        // Propose new theta1
        double proposal1 = currentSample[0] + scale1 * random.nextGaussian();
        proposal1 = wrapAngle(proposal1);

        // Compute new log density
        double proposalLogDensity = computeLogDensity(proposal1, currentSample[1]);

        // Accept/reject
        attempts[0]++;
        if (Math.log(random.nextDouble()) < proposalLogDensity - currentLogDensity) {
            currentSample[0] = proposal1;
            currentLogDensity = proposalLogDensity;
            acceptances[0]++;
        }
    }

    private void updateTheta2() {
        // Propose new theta2
        double proposal2 = currentSample[1] + scale2 * random.nextGaussian();
        proposal2 = wrapAngle(proposal2);

        // Compute new log density
        double proposalLogDensity = computeLogDensity(currentSample[0], proposal2);

        // Accept/reject
        attempts[1]++;
        if (Math.log(random.nextDouble()) < proposalLogDensity - currentLogDensity) {
            currentSample[1] = proposal2;
            currentLogDensity = proposalLogDensity;
            acceptances[1]++;
        }
    }

    private double computeLogDensity(double theta1, double theta2) {
        double logSum = 0;
        for (int i = 0; i < numComponents; i++) {
            logSum += components[i].logPdf(theta1, theta2);
        }
        return logSum;
    }

    private void adaptScales() {
        // Adapt scale1
        if (attempts[0] > 0) {
            double rate1 = (double) acceptances[0] / attempts[0];
            if (rate1 < TARGET_ACCEPTANCE - 0.1) {
                scale1 *= 0.9;
            } else if (rate1 > TARGET_ACCEPTANCE + 0.1) {
                scale1 *= 1.1;
            }
            scale1 = Math.max(0.01, Math.min(2.0, scale1));
        }

        // Adapt scale2
        if (attempts[1] > 0) {
            double rate2 = (double) acceptances[1] / attempts[1];
            if (rate2 < TARGET_ACCEPTANCE - 0.1) {
                scale2 *= 0.9;
            } else if (rate2 > TARGET_ACCEPTANCE + 0.1) {
                scale2 *= 1.1;
            }
            scale2 = Math.max(0.01, Math.min(2.0, scale2));
        }

        // Reset counters
        acceptances[0] = acceptances[1] = 0;
        attempts[0] = attempts[1] = 0;
    }

    private double wrapAngle(double angle) {
        angle = angle % (2 * Math.PI);
        return angle < 0 ? angle + 2 * Math.PI : angle;
    }
}