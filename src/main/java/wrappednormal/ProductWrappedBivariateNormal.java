package wrappednormal;

import java.util.Arrays;
import java.util.Random;

/**
 * A class representing a product of Wrapped Bivariate Normal distributions.
 * The PDF is the product of individual wrapped bivariate normal PDFs:
 * f(θ₁, θ₂) = Π fᵢ(θ₁, θ₂) / Z
 * where Z is the normalization constant.
 *
 * This distribution is useful for conditional sampling and combining multiple
 * angular constraints, where the variance of each component naturally provides
 * its influence (smaller variance = stronger constraint).
 */
public class ProductWrappedBivariateNormal {
    private final WrappedBivariateNormal[] components;
    private final int numComponents;
    private Random random;

    // Cache for normalization constant
    private Double cachedNormalizationConstant = null;
    private int cachedGridSize = 0;

    // Sampler
    private ProductSampler sampler;

    /**
     * Constructs a product of wrapped bivariate normal distributions.
     * Uses Gibbs sampling by default.
     *
     * @param components Array of wrapped bivariate normal distributions
     */
    public ProductWrappedBivariateNormal(WrappedBivariateNormal... components) {
        if (components == null || components.length == 0) {
            throw new IllegalArgumentException("Must have at least one component");
        }

        this.numComponents = components.length;
        this.components = Arrays.copyOf(components, numComponents);
        this.random = new Random();

        // Use Gibbs sampler by default
        this.sampler = new GibbsSampler(this.components);
    }

    /**
     * Constructs a product distribution with a specified random seed.
     *
     * @param seed Random seed for reproducible sampling
     * @param components Array of wrapped bivariate normal distributions
     */
    public ProductWrappedBivariateNormal(long seed, WrappedBivariateNormal... components) {
        this(components);
        this.random = new Random(seed);
        this.sampler.setSeed(seed);
    }

    /**
     * Sets the sampling method to use.
     *
     * @param sampler The ProductSampler implementation to use
     */
    public void setSampler(ProductSampler sampler) {
        this.sampler = sampler;
        if (random != null) {
            sampler.setSeed(random.nextLong());
        }
    }

    /**
     * Gets the current sampler.
     *
     * @return The current ProductSampler
     */
    public ProductSampler getSampler() {
        return sampler;
    }

    /**
     * Creates a Gibbs sampler for this distribution.
     *
     * @return A new GibbsSampler instance
     */
    public ProductSampler createGibbsSampler() {
        return new GibbsSampler(components);
    }

    /**
     * Creates a rejection sampler for this distribution.
     *
     * @return A new RejectionSampler instance
     */
    public ProductSampler createRejectionSampler() {
        return new RejectionSampler(components);
    }

    /**
     * Creates a mixture importance sampler for this distribution.
     *
     * @return A new MixtureImportanceSampler instance
     */
    public ProductSampler createMixtureImportanceSampler() {
        return new MixtureImportanceSampler(components);
    }

    /**
     * Computes the unnormalized probability density function at the given point.
     * This is the product of individual PDFs.
     *
     * @param theta1 First angular coordinate in [0, 2π)
     * @param theta2 Second angular coordinate in [0, 2π)
     * @return The unnormalized probability density at (theta1, theta2)
     */
    public double unnormalizedPdf(double theta1, double theta2) {
        double product = 1.0;

        for (WrappedBivariateNormal component : components) {
            product *= component.pdf(theta1, theta2);

            // Early termination if product becomes zero
            if (product == 0.0) {
                return 0.0;
            }
        }

        return product;
    }

    /**
     * Computes the log of the unnormalized PDF for numerical stability.
     *
     * @param theta1 First angular coordinate in [0, 2π)
     * @param theta2 Second angular coordinate in [0, 2π)
     * @return The log unnormalized probability density at (theta1, theta2)
     */
    public double logUnnormalizedPdf(double theta1, double theta2) {
        double logSum = 0.0;

        for (WrappedBivariateNormal component : components) {
            logSum += component.logPdf(theta1, theta2);
        }

        return logSum;
    }

    /**
     * Estimates the normalization constant using numerical integration.
     * The result is cached for efficiency.
     *
     * @param gridSize Number of grid points in each dimension for integration
     * @return Estimate of the normalization constant
     */
    public double estimateNormalizationConstant(int gridSize) {
        // Return cached value if available and grid size matches
        if (cachedNormalizationConstant != null && cachedGridSize == gridSize) {
            return cachedNormalizationConstant;
        }

        double sum = 0.0;
        double delta = 2 * Math.PI / gridSize;

        for (int i = 0; i < gridSize; i++) {
            for (int j = 0; j < gridSize; j++) {
                double theta1 = i * delta + delta / 2; // Mid-point rule
                double theta2 = j * delta + delta / 2;
                sum += unnormalizedPdf(theta1, theta2);
            }
        }

        cachedNormalizationConstant = sum * delta * delta;
        cachedGridSize = gridSize;

        return cachedNormalizationConstant;
    }

    /**
     * Computes the normalized probability density function at the given point.
     * Uses default grid size of 100 for normalization constant estimation.
     *
     * @param theta1 First angular coordinate in [0, 2π)
     * @param theta2 Second angular coordinate in [0, 2π)
     * @return The normalized probability density at (theta1, theta2)
     */
    public double pdf(double theta1, double theta2) {
        return pdf(theta1, theta2, 100);
    }

    /**
     * Computes the normalized probability density function at the given point.
     *
     * @param theta1 First angular coordinate in [0, 2π)
     * @param theta2 Second angular coordinate in [0, 2π)
     * @param gridSize Grid size for normalization constant estimation
     * @return The normalized probability density at (theta1, theta2)
     */
    public double pdf(double theta1, double theta2, int gridSize) {
        double Z = estimateNormalizationConstant(gridSize);
        return unnormalizedPdf(theta1, theta2) / Z;
    }

    /**
     * Computes the log normalized PDF.
     *
     * @param theta1 First angular coordinate in [0, 2π)
     * @param theta2 Second angular coordinate in [0, 2π)
     * @return The log normalized probability density
     */
    public double logPdf(double theta1, double theta2) {
        return logPdf(theta1, theta2, 100);
    }

    /**
     * Computes the log normalized PDF.
     *
     * @param theta1 First angular coordinate in [0, 2π)
     * @param theta2 Second angular coordinate in [0, 2π)
     * @param gridSize Grid size for normalization constant estimation
     * @return The log normalized probability density
     */
    public double logPdf(double theta1, double theta2, int gridSize) {
        double Z = estimateNormalizationConstant(gridSize);
        return logUnnormalizedPdf(theta1, theta2) - Math.log(Z);
    }

    /**
     * Finds the approximate mode of the distribution using grid search.
     *
     * @param gridSize Size of the grid for searching
     * @return Array containing [theta1_mode, theta2_mode]
     */
    public double[] findMode(int gridSize) {
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

    /**
     * Generates a random sample using the current sampler.
     *
     * @return A 2-element array containing the sampled angles in [0, 2π)
     */
    public double[] sample() {
        return sampler.sample();
    }

    /**
     * Generates multiple samples from the distribution.
     *
     * @param n Number of samples to generate
     * @return An n×2 array of samples
     */
    public double[][] sample(int n) {
        return sampler.sample(n);
    }

    /**
     * Gets the sampling statistics from the current sampler.
     *
     * @return SamplerStatistics object
     */
    public SamplerStatistics getSamplerStatistics() {
        return sampler.getStatistics();
    }

    /**
     * Computes the effective degrees of freedom based on component variances.
     * Components with smaller variance contribute more to constraining the distribution.
     *
     * @return Array of effective weights based on inverse variances
     */
    public double[] getEffectiveWeights() {
        double[] weights = new double[numComponents];
        double sumInvVar = 0.0;

        for (int i = 0; i < numComponents; i++) {
            double variance = components[i].getSigma1() * components[i].getSigma1() +
                    components[i].getSigma2() * components[i].getSigma2();
            weights[i] = 1.0 / variance;
            sumInvVar += weights[i];
        }

        // Normalize
        for (int i = 0; i < numComponents; i++) {
            weights[i] /= sumInvVar;
        }

        return weights;
    }

    /**
     * Returns the number of components in the product.
     */
    public int getNumComponents() {
        return numComponents;
    }

    /**
     * Returns a copy of the component distributions.
     */
    public WrappedBivariateNormal[] getComponents() {
        return Arrays.copyOf(components, numComponents);
    }

    /**
     * Clears the cached normalization constant and resets the sampler.
     */
    public void clearCache() {
        cachedNormalizationConstant = null;
        cachedGridSize = 0;
        sampler.reset();
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder("ProductWrappedBivariateNormal(");
        sb.append("numComponents=").append(numComponents);

        double[] effectiveWeights = getEffectiveWeights();
        sb.append(", effectiveWeights=[");
        for (int i = 0; i < numComponents; i++) {
            if (i > 0) sb.append(", ");
            sb.append(String.format("%.3f", effectiveWeights[i]));
        }
        sb.append("], sampler=").append(sampler.getName());
        sb.append(")");
        return sb.toString();
    }

    /**
     * Example usage demonstrating different samplers.
     */
    public static void main(String[] args) {
        System.out.println("Product of Wrapped Bivariate Normals - Sampler Comparison");
        System.out.println("=========================================================\n");

        // Create components
        WrappedBivariateNormal[] components = {
                new WrappedBivariateNormal(Math.PI/2, Math.PI, 0.2, 0.2, 0.1),
                new WrappedBivariateNormal(Math.PI, Math.PI/2, 0.6, 0.6, 0.2),
                new WrappedBivariateNormal(3*Math.PI/4, 3*Math.PI/4, 0.4, 0.4, 0.0)
        };

        // Create product distribution
        ProductWrappedBivariateNormal product = new ProductWrappedBivariateNormal(components);

        System.out.println("Distribution: " + product);
        System.out.println("\nTesting different samplers:\n");

        // Test each sampler
        ProductSampler[] samplers = {
                product.createGibbsSampler(),
                product.createRejectionSampler(),
                product.createMixtureImportanceSampler()
        };

        for (ProductSampler sampler : samplers) {
            product.setSampler(sampler);
            sampler.setSeed(12345); // For reproducibility

            System.out.println("Using " + sampler.getName());

            long startTime = System.currentTimeMillis();
            double[][] samples = product.sample(100);
            long endTime = System.currentTimeMillis();

            System.out.println("Time for 100 samples: " + (endTime - startTime) + " ms");
            System.out.println("Statistics: " + product.getSamplerStatistics());
            System.out.println();
        }
    }
}