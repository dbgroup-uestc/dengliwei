package dataPrepare.ProxEmbed;

import java.io.FileWriter;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

/**
 * Generate entity features by information from neighbours -- just for symmetric
 */
public class GenerateEntitiesFeatureByTypes {

	private Set<String> types=new HashSet<String>();
	private Map<String,Integer> type2Typeid=new HashMap<String, Integer>();
	private Map<Integer,String> typeid2Type=new HashMap<Integer, String>();

	static String nodes_path=Config.NODES_PATH;     //node文件的路径
	static String edges_path=Config.EDGES_PATH;     //edge文件的路径
	static String entities_feature_file=Config.NODES_FEATURE_SAVE_PATH;		//设置节点特征的保存路径
	static String typeAndTypeIdPath=Config.TYPE_TYPEID_SAVEFILE;		//type与typeid的映射文件的路径
	static double feature_type_value=Config.FEATURE_TYPE_VALUE;			//x中的第一个观察量中的非零值，即是one-hot中的one的具体值
	
	public static void main(String[] args) {
		ReadWholeGraph rwg=new ReadWholeGraph();
		Map<Integer,Node> graph=rwg.readDataFromFile(nodes_path, edges_path, typeAndTypeIdPath);	//利用节点以及边的信息，生成一个图
		GenerateEntitiesFeatureByTypes gefb=new GenerateEntitiesFeatureByTypes();
		gefb.analyseTypes(graph);	//对变量types，type2Typeid，typeid2Type赋值
		gefb.generateFeaturesByGraph(graph, entities_feature_file,feature_type_value);	//利用graph生成特征，并将特征存入entities_feature_file文件中
	}


	/**
	 * Analyse this graph.
	 * @param graph
	 */
	public void analyseTypes(Map<Integer,Node> graph){
		for(Node n:graph.values()){		//循环graph中的所有节点
			types.add(n.getType());		//将节点的type添加到types集合中
			if(!type2Typeid.containsKey(n.getType())){	//如果type2Typeid中没有包含节点n的类型，那么将该类型添加到type2Typeid以及typeid2Type中
				type2Typeid.put(n.getType(), type2Typeid.size());
				typeid2Type.put(typeid2Type.size(), n.getType());
			}	
		}
	}
	
	/**
	 * Generate nodes features.
	 * @param graph
	 * @param saveFile
	 节点特征包含四个观察量：
	 1、节点类型（使用one-hot编码，K维）
	 2、节点的度（标量）
	 3、distribution of neighbours' types：同样是一个K维向量，每一维度都是它邻居是该类型的数量
	 4、邻居类型的熵：利用3来进行计算
	 以下的计算，仅仅是使用了第一个观察量
	 */
	public void generateFeaturesByGraph(Map<Integer,Node> graph,String saveFile,double typeValue){
		int dimension=types.size();		//确定了特征量1以及3的维度
		int nodesNum=graph.size();		//节点的个数
		StringBuilder sb=new StringBuilder();
		String type=null;
		int typeId=0;
		Map<String,Integer> typesNum=new HashMap<String, Integer>();
		FileWriter writer = null;
		try {
			writer = new FileWriter(saveFile);		//打开特征存储（entities_feature_file）文件，准备向其中写数据
			writer.write(nodesNum+" "+dimension+"\r\n");	//首先写入节点数以及维度数
			writer.flush();		//强制写入
			for(Node now:graph.values()){		//循环graph中的节点
				sb.delete( 0, sb.length() );	//清空sb中的内容
				typesNum.clear();				//清空typesNum中的内容
				
				sb.append(now.getId()+" ");		//将当前节点的ID添加到sb之后，以空格分割
				type=now.getType();
				typeId=type2Typeid.get(type);	//获得当前节点的类型，并且利用type2Typeid得到typeID
				
				for(int i=0;i<types.size();i++){	//在所有的类型中循环，如果typeid等于i，这将该位设置为typeValue值，其余值都是0.0
					if(i==typeId){					//即是得到one-hot向量，(0.0,0.0,...,0.0,typeValue,0.0,...,0.0)，存入时是使用空格作为分割
						sb.append(typeValue+" ");	//并且第一个数字为节点id
					}
					else{
						sb.append(0.0+" ");
					}
				}
				
				sb.append("\r\n");
				writer.write(sb.toString());	//将输入LSTM的特征x存入entities_feature_file文件
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
