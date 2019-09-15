package Genetic;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.nio.channels.FileChannel;
import java.util.ArrayList;

import org.jgap.Chromosome;
import org.jgap.Configuration;
import org.jgap.FitnessFunction;
import org.jgap.Gene;
import org.jgap.Genotype;
import org.jgap.IChromosome;
import org.jgap.InvalidConfigurationException;
import org.jgap.Population;
import org.jgap.audit.EvolutionMonitor;
import org.jgap.audit.IEvolutionMonitor;
import org.jgap.event.GeneticEvent;
import org.jgap.event.GeneticEventListener;
import org.jgap.impl.DefaultConfiguration;
import org.jgap.impl.DoubleGene;
import org.jgap.impl.FittestPopulationMerger;
import org.jgap.impl.GABreeder;
import org.jgap.impl.job.SimplePopulationSplitter;

/**
 *  PlayerTrainer but uses multiple threads to train a larger population
 */
public class PlayerTrainerMT {

	public static final int NUM_THREADS = 4; // 4;
	public static final int MAX_EVOLUTION_PERIOD = 1; //40;
	public static final int MAX_EVOLUTION_CYCLES = 6; //5;
	public static final int POPULATION_SIZE = 25;

	public static void main(String[] args) throws Exception {

		backupLog();
		
		SimplePopulationSplitter pSplitter = new SimplePopulationSplitter(NUM_THREADS);
		
		Configuration rootConf = new DefaultConfiguration();
		Gene[] genes = new Gene[7];
		genes[0] = new DoubleGene(rootConf, 0, 2);
		genes[1] = new DoubleGene(rootConf, 0, 2);
		genes[2] = new DoubleGene(rootConf, 0, 2);
		genes[3] = new DoubleGene(rootConf, 0, 2);
		genes[4] = new DoubleGene(rootConf, 0, 2);
		genes[5] = new DoubleGene(rootConf, 0, 2);
		genes[6] = new DoubleGene(rootConf, 0, 2);
		Chromosome sampleChromosome = new Chromosome(rootConf, genes);
		rootConf.setSampleChromosome(sampleChromosome);
		rootConf.setPopulationSize(POPULATION_SIZE * NUM_THREADS);
		rootConf.setFitnessFunction(new PlayerFitnessFunction());

		
		
        BufferedReader in = new BufferedReader(new FileReader("log.txt"));
        Population p = new Population(rootConf);
        String line = null;
        int count = 0;
        while ((line = in.readLine()) != null){
        String[] arr = line.split(" "); genes = new Gene[7]; genes[0] = new
        DoubleGene(rootConf,0, 2);
        genes[0].setAllele(Double.parseDouble(arr[0]));
        genes[1] = new DoubleGene(rootConf, 0, 2);
        genes[1].setAllele(Double.parseDouble(arr[1]));
        genes[2] = new DoubleGene(rootConf, 0, 2);
        genes[2].setAllele(Double.parseDouble(arr[2]));
        genes[3] = new DoubleGene(rootConf, 0, 2);
        genes[3].setAllele(Double.parseDouble(arr[3]));
        genes[4] = new DoubleGene(rootConf, 0, 2);
        genes[4].setAllele(Double.parseDouble(arr[4]));
        genes[5] = new DoubleGene(rootConf, 0, 2);
        genes[5].setAllele(Double.parseDouble(arr[5]));
        genes[6] = new DoubleGene(rootConf, 0, 2);
        genes[6].setAllele(Double.parseDouble(arr[6]));
        Chromosome c = new Chromosome(rootConf, genes);
        p.addChromosome(c);
        count++;
        }
        in.close();
        rootConf.setPopulationSize(count);
        Genotype population = new Genotype(rootConf, p);
		
		
		
		
		
//		Genotype gt = Genotype.randomInitialGenotype(rootConf);
		Population[] populations = pSplitter.split(population.getPopulation());
		
		
		
		
		
		for (int i = 0; i < MAX_EVOLUTION_CYCLES; i++){
            System.out.println("EVOLUTION CYCLE NO. " + i);
			populations = megaEvolve(populations);
			System.out.println("merge");
			Population pop = megaMerge(populations);
			for (int j = 0; j < NUM_THREADS; j++){
				populations[j] = pop;
			}
			
			System.out.println("write");
			updateLog(pop.toChromosomes());
			
			IChromosome best = population.getFittestChromosome();
            Gene[] gene_array = best.getGenes();

			String s = "Fittest chromosome weights and value: ";
            for (int g = 0; g < gene_array.length; g++) {
                s += gene_array[g].getAllele() + " ";
            }
            s += best.getFitnessValue() + " ";
            System.out.println(s);
			System.out.println(best.getFitnessValue());
		}	
	}
	
	public static Population megaMerge(Population[] populations){
		FittestPopulationMerger pMerger = new FittestPopulationMerger();
		Population p = populations[0];
		for (int i = 1; i < NUM_THREADS; i++){
			p = pMerger.mergePopulations(p, populations[i], POPULATION_SIZE);
		}
		return p;
	}
	
	public static Population[] megaEvolve(Population[] populations) throws InvalidConfigurationException{

		Mutex mutex = new Mutex();
		Population[] newPopulations = new Population[NUM_THREADS];
		
		ArrayList<Thread> ts = new ArrayList<Thread>();
		
		for (int i = 0; i < NUM_THREADS; i++){
		    // Construct configuration with unique ID, this is important.
			Configuration.reset(i + "");
		    Configuration conf = new DefaultConfiguration(i + "", "Conf for thread");
		    Gene[] genes = new Gene[7];
			genes[0] = new DoubleGene(conf, 0, 2);
			genes[1] = new DoubleGene(conf, 0, 2);
			genes[2] = new DoubleGene(conf, 0, 2);
			genes[3] = new DoubleGene(conf, 0, 2);
			genes[4] = new DoubleGene(conf, 0, 2);
			genes[5] = new DoubleGene(conf, 0, 2);
			genes[6] = new DoubleGene(conf, 0, 2);
			Chromosome chromosome = new Chromosome(conf, genes);
			conf.setSampleChromosome(chromosome);
			conf.setPopulationSize(POPULATION_SIZE);
			FitnessFunction fitnessFunction = new PlayerFitnessFunction();
			conf.setFitnessFunction(fitnessFunction);

			Genotype genotype = new Genotype(conf, populations[i]);
			
		    final IEvolutionMonitor monitor = new EvolutionMonitor();
	        genotype.setUseMonitor(true);
	        conf.setMonitor(monitor);
	        genotype.setMonitor(monitor);
		    
		    final Thread t1 = new Thread(genotype);
		  
		    GeneticEventListener listener = new GeneticEventListener(){
		    	public void geneticEventFired(GeneticEvent a_firedEvent) {
		    		GABreeder breeder = (GABreeder) a_firedEvent.getSource();
			        int evno = breeder.getLastConfiguration().getGenerationNr();
			        if (evno > MAX_EVOLUTION_PERIOD) {
			            t1.interrupt();
			            monitor.getPopulations();
			            Population p = breeder.getLastPopulation();
			            System.out.println(t1.getName() + " has completed " + evno + " evolution periods.");
			            try {
							mutex.take();
							newPopulations[Integer.parseInt(conf.getId())] = p;
							mutex.release();
						} catch (InterruptedException e) {
							e.printStackTrace();
						}
			        } 
			    }
		    };
		    
		    conf.getEventManager().addEventListener(GeneticEvent.GENOTYPE_EVOLVED_EVENT, listener);
		    
		    ts.add(t1);
		    t1.start();
		}
	
        try {
            for (Thread t: ts) {
                t.join();
            }
        } catch(Exception e) {
            System.out.println("gg");
        }
		
		return newPopulations;
	}
	
	public static void updateLog(IChromosome[] chromosomes) throws IOException{
		backupLog();
		clearLog();
		PrintWriter out = new PrintWriter(new BufferedWriter(
				new FileWriter("log.txt", true)));
		for (int j = 0; j < chromosomes.length; j++) {
			String s = "";
			IChromosome c = chromosomes[j];
			Gene[] gene_array = c.getGenes();
			for (int k = 0; k < gene_array.length; k++) {
				Gene g = gene_array[k];
				s += (double) g.getAllele() + " ";
			}
			s += c.getFitnessValue();
			out.println(s);
		}
		out.close();
	}
	
	public static void clearLog() throws FileNotFoundException{
		PrintWriter out = new PrintWriter("log.txt");
		out.print("");
		out.close();
	}
	
	public static void backupLog() throws IOException {
		FileInputStream srcStream = new FileInputStream("log.txt");
		FileOutputStream destStream = new FileOutputStream("log_old.txt");
		FileChannel src = srcStream.getChannel();
		FileChannel dest = destStream.getChannel();
		dest.transferFrom(src, 0, src.size());
		srcStream.close();
		destStream.close();
	}
}
