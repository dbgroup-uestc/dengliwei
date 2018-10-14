package dataPrepare.ProxEmbed;

import java.io.FileWriter;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

/**
 * Generate entity features by information from neighbours -- just for asymmetric
 */
public class GenerateEntitiesFeaturesByGraph {

	private Set<String> types=new HashSet<String>();
	private Map<String,Integer> type2Typeid=new HashMap<String, Integer>();
	private Map<Integer,String> typeid2Type=new HashMap<Integer, String>();
	static String nodes_path=Config.NODES_PATH;
	static String edges_path=Config.EDGES_PATH;
	static String entities_feature_file=Config.NODES_FEATURE_SAVE_PATH;
	static String typeAndTypeIdPath=Config.TYPE_TYPEID_SAVEFILE;
	static double feature_type_value=Config.FEATURE_TYPE_VALUE;
	
	public static void main(String[] args) {
		ReadWholeGraph rwg=new ReadWholeGraph();
		Map<Integer,Node> graph=rwg.readDataFromFile(nodes_path, edges_path, typeAndTypeIdPath);
		GenerateEntitiesFeaturesByGraph gefb=new GenerateEntitiesFeaturesByGraph();
		gefb.analyseTypes(graph);
		gefb.generateFeaturesByGraph(graph, entities_feature_file,feature_type_value);
	}

	/**
	 * Analyse nodes types
	 */
	public void analyseTypes(Map<Integer,Node> graph){
		for(Node n:graph.values()){
			types.add(n.getType());
			if(!type2Typeid.containsKey(n.getType())){
				type2Typeid.put(n.getType(), type2Typeid.size());
				typeid2Type.put(typeid2Type.size(), n.getType());
			}	
		}
	}
	
	/**
	 * Generate nodes features
	 节点特征包含四个观察量：
	 1、节点类型（使用one-hot编码，K维）
	 2、节点的度（标量）
	 3、distribution of neighbours' types：同样是一个K维向量，每一维度都是它邻居是该类型的数量
	 4、邻居类型的熵：利用3来进行计算
	 */
	public void generateFeaturesByGraph(Map<Integer,Node> graph,String saveFile,double typeValue){
		int dimension=types.size()+1+types.size()+1;
		int nodesNum=graph.size();
		StringBuilder sb=new StringBuilder();
		String type=null;
		int typeId=0;
		double value=0;
		double sum=0;
		Map<String,Integer> typesNum=new HashMap<String, Integer>();
		FileWriter writer = null;
		try {
			writer = new FileWriter(saveFile);
			writer.write(nodesNum+" "+dimension+"\r\n");	//首先写入节点的数量以及节点的维度
			writer.flush();
			for(Node now:graph.values()){
				sb.delete( 0, sb.length() );
				typesNum.clear();
				
				sb.append(now.getId()+" ");			//将节点id作为第一个字符
				type=now.getType();
				typeId=type2Typeid.get(type);
				
				for(int i=0;i<types.size();i++){	//生成节点的第一个特征
					if(i==typeId){
						sb.append(typeValue+" ");
					}
					else{
						sb.append(0.0+" ");
					}
				}
				
				value=now.in_nodes.size();		//生成节点的第二个特征，将节点的入度当做节点的度
				sb.append(Math.log(value+1.0)+" ");	//将该度+1.0，然后取对数，作为节点的第二个特征
				
				for(Node n:now.in_nodes){	//对所有以进入now的节点进行循环
					type=n.getType();		//获得该节点的类型
					if(typesNum.containsKey(type)){				//统计now.in_nodes的节点的类型分布（即是各个类型有多少个）
						typesNum.put(type, typesNum.get(type)+1);
					}
					else{
						typesNum.put(type, 1);
					}
				}
				for(int i=0;i<typeid2Type.size();i++){		//生成一个K维的向量，K是总的类型数，如果now.in_nodes中不含有该类型，那么将其对应的值置为0.0，
					type=typeid2Type.get(i);
					if(typesNum.containsKey(type)){
						sb.append(Math.log(typesNum.get(type)+1)+" ");
					}
					else{
						sb.append(0.0+" ");
					}
				}	//生成节点的第三个特征
				
				value=0;
				sum=0;
				for(int num:typesNum.values()){
					value=(num+0.0)/now.in_nodes.size();	//value=now.in_nodes同一类型的节点的数量/in_nodes的总数
					sum+=-value*Math.log(value);	//计算value的熵
				}//将所有的value的熵相加
				sb.append(sum);		//生成节点的第四个特征
				
				sb.append("\r\n");
				writer.write(sb.toString());	//x的维度是：K+1+K+1
				writer.flush();
			}
		} catch (Exception e) {
			e.printStackTrace();
		}
		finally{
			try {
				if(writer!=null){
					writer.close();
					writer=null;
				}
			} catch (Exception e2) {
				e2.printStackTrace();
			}
		}
	}
}
