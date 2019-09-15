import java.util.Arrays;
import java.util.Collection;
import java.util.LinkedList;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

/**
 * The PlayerSkeleton class implements a utility-based agent that plays the Tetris game.
 * The agent receives the current state of the game as input, which consits of two components:
 * 1. The board position, that is, a binary description of the full/empty status of each square
 * 2. The shape of the current falling object
 * From this state, the agent utilizes a heuristic function comprising a linear weighted sum of 
 * features to evaluate the possible actions and make the best possible action for the current state.
 * 
 * This heuristic function makes use of the following features: 
 * 1. The number of holes present 
 * 2. The number of rows cleared by making the move
 * 3. The height variation - sum of all differences of adjacent column height
 * 4. The maximum column height
 * 5. The sum of all pit depths ( A pit is defined as the difference in height between a column and its two adjacent columns, with a minimum difference of 3)
 * 6. The mean height difference - the average of ALL height differences of each column with the MEAN height.
 * 7. Is the state a lost state from making the move
 * 
 * To determine the weights for each feature in the heuristic function, we used 
 * genetic algorithm to train the AI, treating the set of seven weights as a 
 * chromosome (where each allele corresponds to a weight) and the fitness
 * function computes the fitness value by evaluating the number of rows cleared
 * by the chromosome (the set of weights). Starting with an initial population of 100,
 * we ran the genetic algorithm over many evolutions and chose the fittest chromosome
 * at the end as our weights for the player.
 * 
 * To make our AI play better, we also implement a lookahead for each possible move so
 * that the AI can make better decision for the long term instead of only looking at the
 * outcome of the current set of moves. Indeed, this helped to improve the peformance
 * of our AI.
 * 
 * To make our AI play faster, we implemented multi-threading in 2 key areas:
 * 1. At any state, to split the evaluation of all legal moves for each thread
 * 2. During the lookahead evaluation, we need to account for 4 possible next pieces. 
 *    The evaluation for each piece here is split across multiple threads.
 * This allows the AI to play faster and thus learn faster since learning requires
 * evaluation, which is playing to see the number of rows cleared.
 * 
 */
public class PlayerSkeleton {
    
    // Minimum column height limit before doing lookahead
    public static int LOOKAHEAD_LIMIT = 5;
    public static final int NUM_THREADS = 4;

    public static double NUM_HOLES_WEIGHT;
    public static double COMPLETE_LINES_WEIGHT;
    public static double HEIGHT_VAR_WEIGHT;
    public static double LOST_WEIGHT;
    public static double MAX_HEIGHT_WEIGHT;
    public static double PIT_DEPTH_WEIGHT;
    public static double MEAN_HEIGHT_DIFF_WEIGHT;
    

    // Essentially the same as State.java. Reproduced here so that we can carry
    // out our local search across all possible resulting states given the
    // current state.
    public static class TestState {
        int[][] field;
        int[] top;
        int turn;
        int rowsCleared;
        boolean lost = false;

        public TestState(State s) {
            this.field = cloneField(s.getField());
            this.top = Arrays.copyOf(s.getTop(), s.getTop().length);
            this.turn = s.getTurnNumber();
            this.rowsCleared = s.getRowsCleared();
        }

        public TestState(TestState s) {
            this.field = cloneField(s.field);
            this.top = Arrays.copyOf(s.top, s.top.length);
            this.turn = s.turn;
            this.rowsCleared = s.rowsCleared;
        }

        private int[][] cloneField(int[][] field) {
            int[][] newField = new int[field.length][];
            for (int i = 0; i < newField.length; i++)
                newField[i] = Arrays.copyOf(field[i], field[i].length);
            return newField;
        }

        // returns false if you lose - true otherwise
        public boolean makeMove(int piece, int orient, int slot) {
            // height if the first column makes contact
            int height = top[slot] - pBottom[piece][orient][0];
            // for each column beyond the first in the piece
            for (int c = 1; c < pWidth[piece][orient]; c++) {
                height = Math.max(height, top[slot + c] - pBottom[piece][orient][c]);
            }

            // check if game ended
            if (height + pHeight[piece][orient] >= ROWS) {
                lost = true;
                return false;
            }

            // for each column in the piece - fill in the appropriate blocks
            for (int i = 0; i < pWidth[piece][orient]; i++) {

                // from bottom to top of brick
                for (int h = height + pBottom[piece][orient][i]; h < height + pTop[piece][orient][i]; h++) {
                    field[h][i + slot] = turn;
                }
            }

            // adjust top
            for (int c = 0; c < pWidth[piece][orient]; c++) {
                top[slot + c] = height + pTop[piece][orient][c];
            }

            // check for full rows - starting at the top
            for (int r = height + pHeight[piece][orient] - 1; r >= height; r--) {
                // check all columns in the row
                boolean full = true;
                for (int c = 0; c < COLS; c++) {
                    if (field[r][c] == 0) {
                        full = false;
                        break;
                    }
                }
                // if the row was full - remove it and slide above stuff down
                if (full) {
                    rowsCleared++;
                    // for each column
                    for (int c = 0; c < COLS; c++) {

                        // slide down all bricks
                        for (int i = r; i < top[c]; i++) {
                            field[i][c] = field[i + 1][c];
                        }
                        // lower the top
                        top[c]--;
                        while (top[c] >= 1 && field[top[c] - 1][c] == 0)
                            top[c]--;
                    }
                }
            }
            return true;
        }
    }
    
    // Thread pool for multi-threading player 
    ExecutorService executorService = Executors.newFixedThreadPool(NUM_THREADS);
    
    // Variables to store the best that will be updated by each thread
    public volatile double bestValueSoFar = -1;
    public volatile TestState bestStateSoFar = null;
    public volatile int bestMoveSoFar = 0;

    // implement this function to have a working system
    public int pickMove(State s, int[][] legalMoves) {

        //////////////////////////////////////////////////////////////////////////
        //
        // Applying Multi-Threading to Player
        //
        //////////////////////////////////////////////////////////////////////////

        // Initialize best values
        bestValueSoFar = -1;
        bestStateSoFar = null;
        bestMoveSoFar = 0;
        
        // Initialize multi-threading essentials
        Mutex mutex = new Mutex();
        Collection<Future<?>> tasks = new LinkedList<Future<?>>();
        
        // Request for work to be done by NUM_THREADS threads.
        for (int i=0; i<NUM_THREADS; i++) {
            final int threadNum = i;
            Future<?> future = executorService.submit(new Callable<Object>() {
                public Object call() throws Exception {
                    TestState localBestStateSoFar = null;
                    double localBestValueSoFar = -1;
                    int localBestMoveSoFar = 0;
                
                    // Split the legalMoves for each thread
                    for (int j=threadNum; j<legalMoves.length; j+=NUM_THREADS) {
                        TestState state = new TestState(s);
                        state.makeMove(s.nextPiece, legalMoves[j][ORIENT], legalMoves[j][SLOT]);
                        
                        double value = 0;
                        if (maxHeight(state) > LOOKAHEAD_LIMIT && !state.lost)
                            value = evaluateState(state);
                        else
                            value = evaluateOneLevelLower(state);
                        
                        if (value > localBestValueSoFar || localBestStateSoFar == null) {
                            localBestStateSoFar = state;
                            localBestValueSoFar = value;
                            localBestMoveSoFar = j;
                        }
                    }
                    
                    try {                        
                        // After thread completes, update the best one at a time
                        mutex.take();
                        
                        if (localBestValueSoFar > bestValueSoFar || bestStateSoFar == null) {
                            bestStateSoFar = localBestStateSoFar;
                            bestValueSoFar = localBestValueSoFar;
                            bestMoveSoFar = localBestMoveSoFar;
                        }

                        mutex.release();
                        
                    } catch (InterruptedException e) {
                        e.printStackTrace();
                    }
                    return null;
                }
            });
            
            // Add to list of tasks to check for completion
            tasks.add(future);
        }
        
        // Wait for all the threads to complete their work
        for (Future<?> currTask : tasks) {
            try {
                currTask.get();
            } catch (Throwable thrown) {
                thrown.printStackTrace();
            }
        }

        return bestMoveSoFar;
    }
    
    
    // Multi-threading for deep search lookahead
    private static int MAX_THREADS_FOR_DEEP_SEARCH = 4;
    ExecutorService executorServiceAgain = Executors.newFixedThreadPool(MAX_THREADS_FOR_DEEP_SEARCH);
    
    // Evaluate the value of the given state by going one layer deeper.
    // Given the board position, for each of the N_PIECES of tetrominos,
    // consider all
    // possible placements and rotations, and find the highest heuristic value
    // of all these resultant states of the particular tetromino. Find the
    // average max heuristic value across all N_PIECES tetrominos: this will be
    // the evaluation value for the state
    private double evaluateState(TestState state) {
        
        Collection<Future<Double>> tasks = new LinkedList<Future<Double>>();
        
        // Try multi-thread for this too.
        for (int i=0; i < N_PIECES; i++) {
            final int pieceNumber = i;
            Future<Double> future = executorServiceAgain.submit(new Callable<Double>() {
               public Double call() throws Exception {
                   double maxSoFar = Integer.MIN_VALUE;
                   for (int j=0; j< legalMoves[pieceNumber].length; j++) {
                       TestState lowerState = new TestState(state);
                       lowerState.makeMove(pieceNumber, legalMoves[pieceNumber][j][ORIENT], legalMoves[pieceNumber][j][SLOT]);
                       maxSoFar = Math.max(maxSoFar, evaluateOneLevelLower(lowerState));
                   }
                   return maxSoFar;
               }
            });
            
            tasks.add(future);           
        }
        
        double sumLowerLevel = 0;
        
        for (Future<Double> task : tasks) {
            try {
                sumLowerLevel += task.get();
            } catch (Throwable thrown) {
                thrown.printStackTrace();
            }
        }

        return sumLowerLevel / N_PIECES;
    }

    // Evaluate the state given features to be tested and weights. Apply
    // heuristic function.
    private double evaluateOneLevelLower(TestState state) {

        double h = -numHoles(state) * NUM_HOLES_WEIGHT + numRowsCleared(state) * COMPLETE_LINES_WEIGHT
                + -heightVariationSum(state) * HEIGHT_VAR_WEIGHT + lostStateValue(state) * LOST_WEIGHT
                + -maxHeight(state) * MAX_HEIGHT_WEIGHT + -pitDepthValue(state) * PIT_DEPTH_WEIGHT
                + -meanHeightDiffValue(state) * MEAN_HEIGHT_DIFF_WEIGHT;
        return h;
    }

    /*
     * ===================== Features calculations =====================
     */

    // By default, set the lost state value as -10
    private int lostStateValue(TestState state) {
        return hasLost(state) ? -10 : 0;
    }

    // The height of the highest column in the board
    private static int maxHeight(TestState s) {
        int[] top = s.top;
        int maxSoFar = -1;
        for (int i : top)
            maxSoFar = Math.max(maxSoFar, i);
        return maxSoFar;
    }

    // Holes are defined as all empty cells that are below the top of each
    // column.
    private static int numHoles(TestState s) {
        int[][] field = s.field;
        int sumHoles = 0;
        for (int col = 0; col < COLS; col++) {
            for (int row = 0; row < s.top[col] - 1; row++) {
                if (field[row][col] == 0) {
                    sumHoles++;
                }
            }
        }
        return sumHoles;
    }

    // Number of rows cleared
    private static int numRowsCleared(TestState s) {
        return s.rowsCleared;
    }

    // summing up the differences of adjacent column heights
    private static int heightVariationSum(TestState s) {
        int[] top = s.top;
        int varSum = 0;
        for (int i = 0; i < top.length - 1; i++)
            varSum += Math.abs(top[i] - top[i + 1]);
        return varSum;
    }

    // Returns true if lost
    private static boolean hasLost(TestState s) {
        return s.lost;
    }

    // The sum of all pit depths. A pit is defined as the difference in height
    // between a column and its two adjacent columns, with a minimum difference
    // of 3.
    public double pitDepthValue(TestState s) {
        int[] top = s.top;
        int pitDepthSum = 0;

        int pitColHeight;
        int leftColHeight;
        int rightColHeight;

        // pit depth of first column
        pitColHeight = top[0];
        rightColHeight = top[1];
        int diff = rightColHeight - pitColHeight;
        if (diff > 2) {
            pitDepthSum += diff;
        }

        for (int col = 0; col < State.COLS - 2; col++) {
            leftColHeight = top[col];
            pitColHeight = top[col + 1];
            rightColHeight = top[col + 2];

            int leftDiff = leftColHeight - pitColHeight;
            int rightDiff = rightColHeight - pitColHeight;
            int minDiff = Math.min(leftDiff, rightDiff);
            if (minDiff > 2) {
                pitDepthSum += minDiff;
            }
        }

        // pit depth of last column
        pitColHeight = top[State.COLS - 1];
        leftColHeight = top[State.COLS - 2];
        diff = leftColHeight - pitColHeight;
        if (diff > 2)
            pitDepthSum += diff;

        return pitDepthSum;

    }

    // The mean height difference is the average of ALL height differences of each column with the MEAN height.
    public double meanHeightDiffValue(TestState s) {
        int[] top = s.top;

        int sum = 0;
        for (int height : top)
            sum += height;

        float meanHeight = (float) sum / top.length;

        float avgDiff = 0;
        for (int height : top)
            avgDiff += Math.abs(meanHeight - height);
        return avgDiff / top.length;
    }

    public static void main(String[] args) {

        State s = new State();
        // The optimal set of weights found after 20 evolutions
        double[] weights = {

                1.0673515084146694,
                0.5518373660872153,
                0.13114119880147435,
                1.6682687332623332,
                0.013608946139451739,
                0.3436325190123959,
                0.3589287624556017

        };
        PlayerSkeleton p = new PlayerSkeleton(weights);
        while (!s.lost) {
            s.makeMove(p.pickMove(s, s.legalMoves()));
            if (s.getRowsCleared() % 10000 == 0)
                System.out.println("Rows cleared: " + s.getRowsCleared());     
        }
        p.executorService.shutdown();
        p.executorServiceAgain.shutdown();
        System.out.println("Player has completed " + s.getRowsCleared() + " rows.");
    }

    public PlayerSkeleton(double[] weights) {
        NUM_HOLES_WEIGHT = weights[0];
        COMPLETE_LINES_WEIGHT = weights[1];
        HEIGHT_VAR_WEIGHT = weights[2];
        LOST_WEIGHT = weights[3];
        MAX_HEIGHT_WEIGHT = weights[4];
        PIT_DEPTH_WEIGHT = weights[5];
        MEAN_HEIGHT_DIFF_WEIGHT = weights[6];

    }
    
    /**
     *  Mutex class used for mutual exclusion in multi-threading portions of the 
     *  AI learning and playing.
     */
    public class Mutex {
        private boolean signal = true;

        public synchronized void release() {
            this.signal = true;
            this.notify();
        }

        public synchronized void take() throws InterruptedException{
            while(!this.signal) wait();
            this.signal = false;
        }
    }



    /*
     * =============== Random info copied from State.java ===============
     */
    public static final int COLS = State.COLS;
    public static final int ROWS = State.ROWS;
    public static final int N_PIECES = State.N_PIECES;
    // all legal moves - first index is piece type - then a list of 2-length
    // arrays
    protected static int[][][] legalMoves = new int[N_PIECES][][];

    // indices for legalMoves
    public static final int ORIENT = 0;
    public static final int SLOT = 1;

    // possible orientations for a given piece type
    protected static int[] pOrients = { 1, 2, 4, 4, 4, 2, 2 };

    // the next several arrays define the piece vocabulary in detail
    // width of the pieces [piece ID][orientation]
    protected static int[][] pWidth = { { 2 }, { 1, 4 }, { 2, 3, 2, 3 }, { 2, 3, 2, 3 }, { 2, 3, 2, 3 }, { 3, 2 },
            { 3, 2 } };
    // height of the pieces [piece ID][orientation]
    private static int[][] pHeight = { { 2 }, // square
            { 4, 1 }, // vertical piece
            { 3, 2, 3, 2 }, // L
            { 3, 2, 3, 2 }, //
            { 3, 2, 3, 2 }, // T
            { 2, 3 }, { 2, 3 } };
    private static int[][][] pBottom = { { { 0, 0 } }, { { 0 }, { 0, 0, 0, 0 } },
            { { 0, 0 }, { 0, 1, 1 }, { 2, 0 }, { 0, 0, 0 } }, // L,
            { { 0, 0 }, { 0, 0, 0 }, { 0, 2 }, { 1, 1, 0 } }, { { 0, 1 }, { 1, 0, 1 }, { 1, 0 }, { 0, 0, 0 } },
            { { 0, 0, 1 }, { 1, 0 } }, { { 1, 0, 0 }, { 0, 1 } } };
    private static int[][][] pTop = { { { 2, 2 } }, { { 4 }, { 1, 1, 1, 1 } },
            { { 3, 1 }, { 2, 2, 2 }, { 3, 3 }, { 1, 1, 2 } }, { { 1, 3 }, { 2, 1, 1 }, { 3, 3 }, { 2, 2, 2 } },
            { { 3, 2 }, { 2, 2, 2 }, { 2, 3 }, { 1, 2, 1 } }, { { 1, 2, 2 }, { 3, 2 } }, { { 2, 2, 1 }, { 2, 3 } } };

    // initialize legalMoves
    // legalMoves[piece type][num legal moves][tuple of orient and slot]
    {
        // for each piece type
        for (int i = 0; i < N_PIECES; i++) {
            // figure number of legal moves
            int n = 0;
            for (int j = 0; j < pOrients[i]; j++) {
                // number of locations in this orientation
                n += COLS + 1 - pWidth[i][j];
            }
            // allocate space
            legalMoves[i] = new int[n][2];
            // for each orientation
            n = 0;
            for (int j = 0; j < pOrients[i]; j++) {
                // for each slot
                for (int k = 0; k < COLS + 1 - pWidth[i][j]; k++) {
                    legalMoves[i][n][ORIENT] = j;
                    legalMoves[i][n][SLOT] = k;
                    n++;
                }
            }
        }
    }

}
