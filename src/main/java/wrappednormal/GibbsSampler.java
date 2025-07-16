package wrappednormal;

import java.util.Random;

/**
 * Gibbs sampler with adaptive slice sampling for product of wrapped bivariate normals.
 * This sampler alternates between sampling θ₁|θ₂ and θ₂|θ₁ using slice sampling.
 * Optimized version with adaptive methods to reduce iterations.
 */
public class GibbsSampler implements ProductSampler {
    private final WrappedBivariateNormal[] components;
    private final int numComponents;
    private Random random;
    private final SamplerStatistics stats;

    // Gibbs sampler state
    private boolean initialized = false;
    private double[] currentSample = new double[2];
    private final int gibbsSteps = 5; // Number of Gibbs steps per sample
    private final int burnIn = 100;    // Burn-in iterations

    // Adaptive slice sampling parameters
    private double sliceWidth = 0.5;
    private static final double MIN_WIDTH = 0.01;
    private static final double MAX_WIDTH = 2.0;
    private static final int ADAPT_INTERVAL = 50;

    // Tracking for adaptation
    private int totalStepOuts = 0;
    private int totalShrinkages = 0;
    private int samplesSinceAdapt = 0;

    // Maximum iterations for safety
    private static final int MAX_STEP_OUT = 10;  // Much reduced from 100
    private static final int MAX_SHRINK = 20;    // Much reduced from 100

    public GibbsSampler(WrappedBivariateNormal[] components) {
        this.components = components;
        this.numComponents = components.length;
        this.random = new Random();
        this.stats = new SamplerStatistics();
    }

    @Override
    public String getName() {
        return "Gibbs Sampler (Adaptive Slice Sampling)";
    }

    @Override
    public String getDescription() {
        return "Alternates between sampling θ₁|θ₂ and θ₂|θ₁ using adaptive slice sampling. " +
                "Uses doubling for step-out and adaptive width adjustment.";
    }

    @Override
    public void setSeed(long seed) {
        this.random = new Random(seed);
    }

    @Override
    public void reset() {
        initialized = false;
        stats.reset();
        sliceWidth = 0.5;
        totalStepOuts = 0;
        totalShrinkages = 0;
        samplesSinceAdapt = 0;
    }

    @Override
    public SamplerStatistics getStatistics() {
        return stats;
    }

    @Override
    public double[] sample() {
        long startTime = System.nanoTime();

        // Initialize on first use
        if (!initialized) {
            initialize();
        }

        // Run Gibbs sampling steps
        for (int step = 0; step < gibbsSteps; step++) {
            currentSample[0] = sampleTheta1GivenTheta2(currentSample[1]);
            currentSample[1] = sampleTheta2GivenTheta1(currentSample[0]);
        }

        // Adapt slice width periodically
        samplesSinceAdapt++;
        if (samplesSinceAdapt >= ADAPT_INTERVAL) {
            adaptSliceWidth();
            samplesSinceAdapt = 0;
        }

        long timeTaken = (System.nanoTime() - startTime) / 1_000_000; // Convert to ms
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
        stats.recordBatch(0, totalTime - n * stats.getTotalTime() / Math.max(1, stats.getTotalSamples()), 0);

        return samples;
    }

    private void initialize() {
        // Quick initialization - start from a random component's mean
        int randomComp = random.nextInt(numComponents);
        currentSample = new double[2];
        currentSample[0] = components[randomComp].getMu1();
        currentSample[1] = components[randomComp].getMu2();

        // Add some noise
        currentSample[0] = wrapAngle(currentSample[0] + 0.1 * random.nextGaussian());
        currentSample[1] = wrapAngle(currentSample[1] + 0.1 * random.nextGaussian());

        // Run burn-in
        for (int i = 0; i < burnIn; i++) {
            currentSample[0] = sampleTheta1GivenTheta2(currentSample[1]);
            currentSample[1] = sampleTheta2GivenTheta1(currentSample[0]);
        }

        initialized = true;
    }

    private void adaptSliceWidth() {
        // Adapt based on average step-outs and shrinkages
        double avgStepOuts = (double) totalStepOuts / (2 * ADAPT_INTERVAL * gibbsSteps);
        double avgShrinkages = (double) totalShrinkages / (2 * ADAPT_INTERVAL * gibbsSteps);

        // If we're stepping out too much, increase width
        if (avgStepOuts > 1.5) {
            sliceWidth = Math.min(sliceWidth * 1.5, MAX_WIDTH);
        }
        // If we're not stepping out much but shrinking a lot, decrease width
        else if (avgStepOuts < 0.5 && avgShrinkages > 3) {
            sliceWidth = Math.max(sliceWidth * 0.8, MIN_WIDTH);
        }

        // Reset counters
        totalStepOuts = 0;
        totalShrinkages = 0;
    }

    private double logUnnormalizedPdf(double theta1, double theta2) {
        double logSum = 0;
        for (WrappedBivariateNormal comp : components) {
            logSum += comp.logPdf(theta1, theta2);
        }
        return logSum;
    }

    /**
     * Efficient step-out using doubling procedure
     */
    private double[] stepOut(double x0, double logU, boolean isTheta1, double fixedValue) {
        double lower = x0 - sliceWidth * random.nextDouble();
        double upper = lower + sliceWidth;

        // Randomized doubling limits
        int j = random.nextInt(MAX_STEP_OUT);
        int k = MAX_STEP_OUT - 1 - j;

        // Expand lower bound
        int actualStepOuts = 0;
        while (j > 0) {
            double wrappedLower = wrapAngle(lower);
            double logDensity = isTheta1 ?
                    logUnnormalizedPdf(wrappedLower, fixedValue) :
                    logUnnormalizedPdf(fixedValue, wrappedLower);

            if (logDensity <= logU) break;

            lower -= sliceWidth;
            j--;
            actualStepOuts++;
        }

        // Expand upper bound
        while (k > 0) {
            double wrappedUpper = wrapAngle(upper);
            double logDensity = isTheta1 ?
                    logUnnormalizedPdf(wrappedUpper, fixedValue) :
                    logUnnormalizedPdf(fixedValue, wrappedUpper);

            if (logDensity <= logU) break;

            upper += sliceWidth;
            k--;
            actualStepOuts++;
        }

        totalStepOuts += actualStepOuts;

        return new double[]{lower, upper};
    }

    /**
     * Shrink interval and sample with early termination
     */
    private double shrinkAndSample(double[] interval, double current, double logU,
                                   boolean isTheta1, double fixedValue) {
        double lower = interval[0];
        double upper = interval[1];

        int shrinkages = 0;
        for (int i = 0; i < MAX_SHRINK; i++) {
            // Check if interval is too small
            if (upper - lower < 1e-10) {
                totalShrinkages += shrinkages;
                return current;
            }

            double proposal = lower + random.nextDouble() * (upper - lower);
            double wrappedProposal = wrapAngle(proposal);

            double logDensity = isTheta1 ?
                    logUnnormalizedPdf(wrappedProposal, fixedValue) :
                    logUnnormalizedPdf(fixedValue, wrappedProposal);

            if (logDensity >= logU) {
                totalShrinkages += shrinkages;
                return wrappedProposal;
            }

            // Shrink interval
            if (proposal < current) {
                lower = proposal;
            } else {
                upper = proposal;
            }
            shrinkages++;
        }

        totalShrinkages += shrinkages;
        return current; // Failed to find acceptable point
    }

    private double sampleTheta1GivenTheta2(double theta2) {
        // Current log density
        double currentLogDensity = logUnnormalizedPdf(currentSample[0], theta2);

        // Slice level
        double logU = Math.log(random.nextDouble()) + currentLogDensity;

        // Step out
        double[] interval = stepOut(currentSample[0], logU, true, theta2);

        // Shrink and sample
        return shrinkAndSample(interval, currentSample[0], logU, true, theta2);
    }

    private double sampleTheta2GivenTheta1(double theta1) {
        // Current log density
        double currentLogDensity = logUnnormalizedPdf(theta1, currentSample[1]);

        // Slice level
        double logU = Math.log(random.nextDouble()) + currentLogDensity;

        // Step out
        double[] interval = stepOut(currentSample[1], logU, false, theta1);

        // Shrink and sample
        return shrinkAndSample(interval, currentSample[1], logU, false, theta1);
    }

    private double wrapAngle(double angle) {
        // More efficient wrapping
        angle = angle % (2 * Math.PI);
        return angle < 0 ? angle + 2 * Math.PI : angle;
    }
}