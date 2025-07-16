package wrappednormal;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.BeforeEach;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Unit tests for ProductWrappedBivariateNormal class.
 */
public class ProductWrappedBivariateNormalTest {

    private WrappedBivariateNormal wbn1;
    private WrappedBivariateNormal wbn2;
    private WrappedBivariateNormal wbn3;
    private ProductWrappedBivariateNormal product;
    private static final double EPSILON = 1e-6;

    @BeforeEach
    public void setUp() {
        // Create components with different variances
        wbn1 = new WrappedBivariateNormal(Math.PI/2, Math.PI, 0.2, 0.2, 0.1);      // tight
        wbn2 = new WrappedBivariateNormal(Math.PI, Math.PI/2, 0.6, 0.6, 0.2);      // loose
        wbn3 = new WrappedBivariateNormal(3*Math.PI/4, 3*Math.PI/4, 0.4, 0.4, 0); // medium

        // Create product
        product = new ProductWrappedBivariateNormal(wbn1, wbn2, wbn3);
    }

    @Test
    public void testConstructorValidation() {
        // Test null components
        assertThrows(IllegalArgumentException.class, () -> {
            new ProductWrappedBivariateNormal((WrappedBivariateNormal[]) null);
        });

        // Test empty components
        assertThrows(IllegalArgumentException.class, () -> {
            new ProductWrappedBivariateNormal(new WrappedBivariateNormal[0]);
        });
    }

    @Test
    public void testSingleComponent() {
        // Product with single component
        ProductWrappedBivariateNormal single = new ProductWrappedBivariateNormal(wbn1);

        double theta1 = Math.PI / 3;
        double theta2 = 2 * Math.PI / 3;

        // Unnormalized PDFs should be equal for single component
        assertEquals(wbn1.pdf(theta1, theta2),
                single.unnormalizedPdf(theta1, theta2),
                EPSILON);
    }

    @Test
    public void testUnnormalizedPdfProperties() {
        // PDF should be non-negative
        for (double theta1 = 0; theta1 < 2 * Math.PI; theta1 += Math.PI / 4) {
            for (double theta2 = 0; theta2 < 2 * Math.PI; theta2 += Math.PI / 4) {
                assertTrue(product.unnormalizedPdf(theta1, theta2) >= 0);
            }
        }
    }

    @Test
    public void testLogPdf() {
        double theta1 = Math.PI;
        double theta2 = Math.PI;

        // Log of product should equal sum of logs
        double logProduct = product.logUnnormalizedPdf(theta1, theta2);
        double sumLogs = wbn1.logPdf(theta1, theta2) +
                wbn2.logPdf(theta1, theta2) +
                wbn3.logPdf(theta1, theta2);

        assertEquals(sumLogs, logProduct, EPSILON);
    }

    @Test
    public void testNormalizationConstant() {
        // Normalization constant should be positive
        double Z = product.estimateNormalizationConstant(50);
        assertTrue(Z > 0);

        // Test caching
        double Z2 = product.estimateNormalizationConstant(50);
        assertEquals(Z, Z2, 0); // Should be exactly equal due to caching

        // Different grid size should give different result
        double Z3 = product.estimateNormalizationConstant(100);
        assertNotEquals(Z, Z3);
    }

    @Test
    public void testNormalizedPdf() {
        // Integral of normalized PDF should be approximately 1
        int gridSize = 100;
        double sum = 0.0;
        double delta = 2 * Math.PI / gridSize;

        for (int i = 0; i < gridSize; i++) {
            for (int j = 0; j < gridSize; j++) {
                double theta1 = i * delta + delta / 2;
                double theta2 = j * delta + delta / 2;
                sum += product.pdf(theta1, theta2, gridSize) * delta * delta;
            }
        }

        assertEquals(1.0, sum, 0.01); // Allow 1% error
    }

    @Test
    public void testFindMode() {
        // Mode should be a valid point
        double[] mode = product.findMode(50);
        assertEquals(2, mode.length);
        assertTrue(mode[0] >= 0 && mode[0] < 2 * Math.PI);
        assertTrue(mode[1] >= 0 && mode[1] < 2 * Math.PI);

        // PDF at mode should be relatively high
        double pdfAtMode = product.unnormalizedPdf(mode[0], mode[1]);
        double pdfElsewhere = product.unnormalizedPdf(0, 0);
        assertTrue(pdfAtMode >= pdfElsewhere);
    }

    @Test
    public void testSampling() {
        // Test that samples are in valid range
        double[][] samples = product.sample(100);

        for (double[] sample : samples) {
            assertEquals(2, sample.length);
            assertTrue(sample[0] >= 0 && sample[0] < 2 * Math.PI);
            assertTrue(sample[1] >= 0 && sample[1] < 2 * Math.PI);
        }
    }

    @Test
    public void testSamplingWithSeed() {
        // Test reproducibility with seed
        ProductWrappedBivariateNormal product1 = new ProductWrappedBivariateNormal(12345L, wbn1, wbn2);
        ProductWrappedBivariateNormal product2 = new ProductWrappedBivariateNormal(12345L, wbn1, wbn2);

        double[] sample1 = product1.sample();
        double[] sample2 = product2.sample();

        assertArrayEquals(sample1, sample2, EPSILON);
    }

    @Test
    public void testEffectiveWeights() {
        // Component with smallest variance should have highest effective weight
        double[] weights = product.getEffectiveWeights();
        assertEquals(3, weights.length);

        // Sum should be 1
        double sum = 0;
        for (double w : weights) {
            sum += w;
        }
        assertEquals(1.0, sum, EPSILON);

        // wbn1 has smallest variance, should have highest weight
        assertTrue(weights[0] > weights[1]); // tight > loose
        assertTrue(weights[0] > weights[2]); // tight > medium
        assertTrue(weights[2] > weights[1]); // medium > loose
    }

    @Test
    public void testProductBehavior() {
        // Product should be more concentrated than individual components
        double theta1 = 3 * Math.PI / 4;
        double theta2 = 3 * Math.PI / 4;

        // Near the mode of wbn3
        double productPdf = product.unnormalizedPdf(theta1, theta2);

        // Far from all modes
        double farTheta1 = 0;
        double farTheta2 = 0;
        double productPdfFar = product.unnormalizedPdf(farTheta1, farTheta2);

        // Product should have stronger peak
        assertTrue(productPdf / productPdfFar > wbn3.pdf(theta1, theta2) / wbn3.pdf(farTheta1, farTheta2));
    }

    @Test
    public void testGetters() {
        assertEquals(3, product.getNumComponents());

        WrappedBivariateNormal[] components = product.getComponents();
        assertEquals(3, components.length);

        // Check that we get copies, not references
        components[0] = null;
        assertNotNull(product.getComponents()[0]);
    }

    @Test
    public void testClearCache() {
        // Estimate normalization constant
        double Z1 = product.estimateNormalizationConstant(50);

        // Clear cache
        product.clearCache();

        // Should recompute (might get slightly different result due to randomness in integration)
        // But with same grid size should be same
        double Z2 = product.estimateNormalizationConstant(50);
        assertEquals(Z1, Z2, EPSILON);
    }

    @Test
    public void testToString() {
        String str = product.toString();
        assertTrue(str.contains("ProductWrappedBivariateNormal"));
        assertTrue(str.contains("numComponents=3"));
        assertTrue(str.contains("effectiveWeights="));
    }
}