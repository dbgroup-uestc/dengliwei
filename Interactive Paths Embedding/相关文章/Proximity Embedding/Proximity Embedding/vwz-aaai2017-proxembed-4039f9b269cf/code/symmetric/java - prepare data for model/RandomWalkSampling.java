package dataPrepare.ProxEmbed;

import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;


/**
 * Generate samplings by random walk samplings.
 * 
 * Procedure:
 * 1.Read the whole graph
 * 2.Generate samplings by random walk.
 */
public class RandomWalkSampling {

	/**
	 * Random number generator
	 */
	private Random random=new Random(123);      //定义随机数生成器，使用种子为123的Random对象
	//当调用random.nextInt()的时候会生成随机数
    
	static String nodesPath=Config.NODES_PATH;     //获取节点、边路径的文件位置
	static String edgesPath=Config.EDGES_PATH;
	static String savePath=Config.SAVE_PATH_FOR_RANDOMWALK_SAMPLINGS;   //设置random walk采样的存储路径
	static int K=Config.SAMPLING_TIMES_PER_NODE;    //获取每个节点的采样次数
	static int L=Config.SAMPLING_LENGTH_PER_PATH;   //获取每条路径的采样长度
	static String typeAndTypeIdPath=Config.TYPE_TYPEID_SAVEFILE;        //类型和类型id的映射文件的保存位置
	static int shortest_path_length=Config.SHORTEST_LENGTH_FOR_SAMPLING;    //最短路径长度
	
	public static void main(String[] args) {
		ReadWholeGraph rwg=new ReadWholeGraph();
		//1.Read the whole graph
		Map<Integer,Node> data=rwg.readDataFromFile(
				nodesPath, 
				edgesPath, 
				typeAndTypeIdPath);     //从文件中读取数据，返回的是Map<Integer,Node> data=new HashMap<Integer,Node>();
		//2.Generate samplings by random walk.
		RandomWalkSampling crws=new RandomWalkSampling();
		crws.randomWalkSampling(data, K, L, savePath);  //data:nodeid与node的映射图，K:每个节点的采样次数，L:每条路径的采样长度，savePath:采样的存储路径
	}

	/**
	 * Generate samplings by random walk.
	 * @param data
	 * @param k
	 * @param l
	 * @param pathsFile
	 */
	public void randomWalkSampling(Map<Integer,Node> data,int k,int l,String pathsFile){
		List<Node> path=null;
		FileWriter writer=null;
		StringBuilder sb=new StringBuilder();   //类似与String类型，只是对它的操作要比String要快
		try {
			writer=new FileWriter(pathsFile);   //打开pathsFile文件，用来向其中写入path
		} catch (IOException e) {
			e.printStackTrace();
		}
		for(Node node:data.values()){   //data.values得到MAP中的value的集合
			for(int i=0;i<k;i++){       //对每个节点采样K次
				path=randomWalkPath(node,l,data);   //每条路径的采样长度的l，对节点进行采样，返回的path是一个Node类型的list
				if(path.size()<shortest_path_length){   //如果所采样的路径的长度小于最小长度，那么忽略该采样
					continue;                           //考虑情况，存在随机行走的到了某一node，然而该node的出度为0，那么采样就会停止，如果此时的path小于shortest path length，这该路径会被舍弃
				}
				sb.delete( 0, sb.length() );    //删除从第一个字母到最后一个字母，即是清空stringbuilder sb
				for(int j=0;j<path.size();j++){
					sb.append(path.get(j).getId()+" "); //对path中的每个节点，获取其node id，并将其一次增加到字符串sb上，以空格作为分割符
				}	//存入的path类似于：1 2 3 4 5 ；即是所采样的路径是1->2->3->4->5
				sb.append("\r\n");  //添加完成之后增加回车符与换行符，这是windows系统中的换行
				try {
					writer.write(sb.toString());    //sb.toString是调用对象的toString方法来获得string，其实不调用也是一样的
					writer.flush(); //清空缓冲区，强制写入
				} catch (IOException e) {
					e.printStackTrace();
				}
			}
			
		}
	}
	
	/**
	 * Generate a path by random walk.
	 * @param start
	 * @param l
	 * @param data
	 * @return
	 */
	private List<Node> randomWalkPath(Node start,int l, Map<Integer,Node> data){
		List<Node> path=new ArrayList<Node>(l+1);   //path列表用来存储节点序列，路径长度为l，意味着有l+1个节点
		path.add(start);    //从传入的节点开始
		Node now=start;
		Set<Integer> types_set=new HashSet<Integer>();
		List<Integer> types=new ArrayList<Integer>();
		Map<Integer,List<Integer>> neighbours=new HashMap<Integer, List<Integer>>();
		int type=-1;
		List<Integer> list=null;
		for(int i=0;i<l;i++){   
			if(now.out_nodes.size()==0){
				break;      //如果当前节点是没有下一个节点了，则退出循环
			}
			types_set.clear();  //清空types_set集合中的元素，types_set用来存放now节点的邻居节点的所有类型的id
			types.clear();      //清空types集合中的元素
			neighbours.clear(); //清空neighbours集合中的元素
			for(Node n:now.out_nodes){  //这个for循环是将now节点的所有邻居存入neighbours中
				types_set.add(n.getTypeId());   //将now节点的下一个节点的类型id增加到types_set集合中
				if(neighbours.containsKey(n.getTypeId())){  //如果neighbours中含有与now节点下一个节点相同的类型
					neighbours.get(n.getTypeId()).add(n.getId());   //如果已经有同样的类型，那么将新增的节点id添加到其对应的map的value上
                    //需要注意到neighbours的key-value对中，value的类型是一个整数的列表
				}
				else{       //如果neighbours中不含有与now节点下一个节点相同类型的节点
					List<Integer> ids=new ArrayList<Integer>();
					ids.add(n.getId());
					neighbours.put(n.getTypeId(), ids); //则将当前节点的类型以及ID存入neighbours
				}
			}
			types.addAll(types_set);    //将types_set集合中的所有元素添加到types列表中
			type=types.get(random.nextInt(types.size()));   //随机选择一个type
			list=neighbours.get(type);      //获得该type的节点的节点id的列表
			now=data.get(list.get(random.nextInt(list.size())));    //从这些节点id中随机选择一个节点作为下一次的now节点
			path.add(now);  //将所选择的节点加入到path中
		}
		return path;
	}
}
