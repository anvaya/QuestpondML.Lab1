namespace QuestpondML.Lab1
{
    internal class Program
    {
        /// <summary>
        /// Author: Mrugendra (Yogesh) Bhure.
        /// Questpond ML Tab Experiments.
        /// </summary>
        /// <param name="args"></param>
        static void Main(string[] args)
        {            
            //Labs.NiftyEstimator.RunSsa();
            //Labs.NiftyEstimator.RunAutoMLExperiment();            
            Labs.NiftyEstimator.RunSVR();
            //Labs.NiftyEstimator.RunReplWithFastForrest();
            
            Console.WriteLine("Press any key to exit...");
            Console.ReadKey(false);
        }        
    }
}
