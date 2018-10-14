package dataPrepare.ProxEmbed;

import java.io.FileInputStream;
import java.util.Properties;

/**
 * Parameters for ProxEmbed to prepare data for training.
 */
public class Config {

	/**
	 * The main dataset directory.
	 * Eg, "D:/dataset/icde2016/dataset/linkedin/" means the directory of linkedin.
	 */
	public static String MAIN_DIR="D:/test/test/toydata/linkedin/"; //数据的主文件目录
	/**
	 * Path of nodes file.
	 */
	public static String NODES_PATH=MAIN_DIR+"graph.node";          //节点文件路径
	/**
	 * Path of edges file.
	 */
	public static String EDGES_PATH=MAIN_DIR+"graph.edge";          //边文件路径
	/**
	 * The result file of random walk sampling.
	 */
	public static String SAVE_PATH_FOR_RANDOMWALK_SAMPLINGS=MAIN_DIR+"randomWalkSamplingPaths";     //randomwalk结果的保存路径
	/**
	 * The file which contains the map relation of type and typeid.
	 */
	public static String TYPE_TYPEID_SAVEFILE=MAIN_DIR+"typeAndTypeIDSavePath";     //类型和类型id的映射文件的保存位置
	/**
	 * The file which contains the nodes features.
	 */
	public static String NODES_FEATURE_SAVE_PATH=MAIN_DIR+"nodesFeatures";          //节点特征的保存位置
	/**
	 * Truncate sub-paths from samplings by this type.
	 */
	public static String TRUNCATED_TYPE_NAME="user";                //从采样中确定要要截断的子路径的类型
	/**
	 * Sub-paths save path.
	 */
	public static String SUBPATHS_SAVE_PATH=MAIN_DIR+"subpathsSaveFile";
	/**
	 * The longest length for sampling to truncate sub-paths.
	 */
	public static int LONGEST_ANALYSE_LENGTH_FOR_SAMPLING=20;
	/**
	 * Longest length for sub-paths
	 */
	public static int LONGEST_LENGTH_FOR_SUBPATHS=5;                //最长的子路径的长度
	/**
	 * The shortest length for each path in sampling results.
	 */
	public static int SHORTEST_LENGTH_FOR_SAMPLING=0;               //最短的路径长度
	/**
	 * Sampling times for per node in random walk sampling.
	 */
	public static int SAMPLING_TIMES_PER_NODE=5;                    //每个节点的random walk次数
	/**
	 * Sampling length for per node in random walk sampling.
	 */
	public static int SAMPLING_LENGTH_PER_PATH=5;           //每条路径的采样长度
	/**
	 * When generate user features by neighbours' information, the value we set for type information when this node belongs to this kind of type.
	 */
	public static double FEATURE_TYPE_VALUE=1.0;
	
	//initialize
	static{

		Properties prop = new Properties();
		FileInputStream in=null;
		try {
			//The path of properties file
			in = new FileInputStream("/usr/lzmExperiment/path2vec/paramsSet/javaParams.properties");
			prop.load(in);
			
			MAIN_DIR=prop.getProperty("MAIN_DIR");
			NODES_PATH=MAIN_DIR+prop.getProperty("NODES_PATH");
			EDGES_PATH=MAIN_DIR+prop.getProperty("EDGES_PATH");
			SAVE_PATH_FOR_RANDOMWALK_SAMPLINGS=MAIN_DIR+prop.getProperty("SAVE_PATH_FOR_RANDOMWALK_SAMPLINGS");
			TYPE_TYPEID_SAVEFILE=MAIN_DIR+prop.getProperty("TYPE_TYPEID_SAVEFILE");
			NODES_FEATURE_SAVE_PATH=MAIN_DIR+prop.getProperty("NODES_FEATURE_SAVE_PATH");
			TRUNCATED_TYPE_NAME=prop.getProperty("TRUNCATED_TYPE_NAME");
			SUBPATHS_SAVE_PATH=MAIN_DIR+prop.getProperty("SUBPATHS_SAVE_PATH");
			LONGEST_ANALYSE_LENGTH_FOR_SAMPLING=Integer.parseInt(prop.getProperty("LONGEST_ANALYSE_LENGTH_FOR_SAMPLING"));
			LONGEST_LENGTH_FOR_SUBPATHS=Integer.parseInt(prop.getProperty("LONGEST_LENGTH_FOR_SUBPATHS"));
			SHORTEST_LENGTH_FOR_SAMPLING=Integer.parseInt(prop.getProperty("SHORTEST_LENGTH_FOR_SAMPLING"));
			SAMPLING_TIMES_PER_NODE=Integer.parseInt(prop.getProperty("SAMPLING_TIMES_PER_NODE"));
			SAMPLING_LENGTH_PER_PATH=Integer.parseInt(prop.getProperty("SAMPLING_LENGTH_PER_PATH"));
			FEATURE_TYPE_VALUE=Double.parseDouble(prop.getProperty("FEATURE_TYPE_VALUE"));
			
			//Print these parameters
			System.out.println("Java parameters is :");
			System.out.println("MAIN_DIR : "+MAIN_DIR);
			System.out.println("NODES_PATH : "+NODES_PATH);
			System.out.println("EDGES_PATH : "+EDGES_PATH);
			System.out.println("SAVE_PATH_FOR_RANDOMWALK_SAMPLINGS : "+SAVE_PATH_FOR_RANDOMWALK_SAMPLINGS);
			System.out.println("TYPE_TYPEID_SAVEFILE : "+TYPE_TYPEID_SAVEFILE);
			System.out.println("NODES_FEATURE_SAVE_PATH : "+NODES_FEATURE_SAVE_PATH);
			System.out.println("TRUNCATED_TYPE_NAME : "+TRUNCATED_TYPE_NAME);
			System.out.println("SUBPATHS_SAVE_PATH : "+SUBPATHS_SAVE_PATH);
			System.out.println("LONGEST_ANALYSE_LENGTH_FOR_SAMPLING : "+LONGEST_ANALYSE_LENGTH_FOR_SAMPLING);
			System.out.println("LONGEST_LENGTH_FOR_SUBPATHS : "+LONGEST_LENGTH_FOR_SUBPATHS);
			System.out.println("SHORTEST_LENGTH_FOR_SAMPLING : "+SHORTEST_LENGTH_FOR_SAMPLING);
			System.out.println("SAMPLING_TIMES_PER_NODE : "+SAMPLING_TIMES_PER_NODE);
			System.out.println("SAMPLING_LENGTH_PER_PATH : "+SAMPLING_LENGTH_PER_PATH);
			System.out.println("FEATURE_TYPE_VALUE : "+FEATURE_TYPE_VALUE);
		} catch (Exception e) {
			e.printStackTrace();
		}
		finally{
			try {
				if(in!=null){
					in.close();
					in=null;
				}
			} catch (Exception e2) {
				e2.printStackTrace();
			}
		}
	
	}
}
