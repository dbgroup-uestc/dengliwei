package dataPrepare.ProxEmbed;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.HashMap;
import java.util.Map;

/**
 * Read the while graph, and then save the info into Map<Integer,Node>
 */
public class ReadWholeGraph {

	static Map<Integer,String> typeid2Type=new HashMap<Integer, String>();  //定义映射，将typeid映射为type
	static Map<String,Integer> type2Typeid=new HashMap<String, Integer>();  //定义映射，将type映射为typeid

	/**
	 * Read whole graph info
	 * @param nodesPath
	 * @param edgesPath
	 * @param typeAndTypeIdPath
	 * @return
	 */
	public Map<Integer,Node> readDataFromFile(String nodesPath,String edgesPath,String typeAndTypeIdPath){  //Node定义在Node.java文件中
		Map<Integer,Node> data=new HashMap<Integer,Node>();     //新建一个映射，将typeid映射为Node
		BufferedReader br=null;         //缓冲流读取器，用于读取nodes文件
		String[] arr=null;
		Node node=null;
		try {   //这个try catch块是用来读取node的信息，type，typeId，id，并且生成映射type2Typeid，typeid2Type，生成节点id到节点的映射
			br = new BufferedReader(new InputStreamReader(new FileInputStream(nodesPath), "UTF-8"));    //读取nodes文件
			String temp = null;
			while ((temp = br.readLine()) != null ) {
				temp=temp.trim();       //去掉字符串首、尾空格
				if(temp.length()>0){    //防止读取空行
					arr=temp.split("\t");   //按照‘tab’进行分割，数据格式：第一项是节点id，第二项是类型，第三项是内容；在整个程序中，第三项没有使用
					node=new Node();
					node.setId(Integer.parseInt(arr[0]));   //利用读取的节点id与节点类型来初始化Node
					node.setType(arr[1]);
					if(type2Typeid.containsKey(arr[1])){    //如果type2Typeid中没有该类型，则将该类型添加到该映射中去，并且设置节点的typeid
						node.setTypeId(type2Typeid.get(arr[1]));    //增加节点属性，TypeId
					}
					else{
						type2Typeid.put(arr[1], type2Typeid.size());    //  map类型利用put将存入映射关系
						typeid2Type.put(typeid2Type.size(), arr[1]);
						node.setTypeId(type2Typeid.get(arr[1]));
					}
					data.put(Integer.parseInt(arr[0]), node);
				}
			}
		} catch (Exception e2) {
			e2.printStackTrace();
		}
		finally{
			try {
				if(br!=null){
					br.close();
					br=null;
				}
			} catch (IOException e) {
				e.printStackTrace();
			}
		}
		int start=0;
		int end=0;
		Node startNode=null;
		Node endNode=null;
		try {   //读取edge信息，并将这些信息分别放入in_nodes、out_nodes、in_ids、in_nodes中
			br = new BufferedReader(new InputStreamReader(new FileInputStream(edgesPath), "UTF-8"));    //读取edges文件，这个文件中的数据是由开始节点指向结束节点的有向边
			String temp = null;
			while ((temp = br.readLine()) != null ) {
				temp=temp.trim();
				if(temp.length()>0){
					arr=temp.split("\t");
					start=Integer.parseInt(arr[0]); //parseInt解析字符串，将字符串转化为相应的整数，默认是10进制
					end=Integer.parseInt(arr[1]);
					startNode=data.get(start);      //从节点id到节点的映射中，根据节点id获取节点
					endNode=data.get(end);          //从节点id到节点的映射中，根据节点id获取节点
					startNode.out_ids.add(end);     //增加输出节点id
					startNode.out_nodes.add(endNode);   //增加输出节点
					endNode.in_ids.add(start);      //增加输入节点id
					endNode.in_nodes.add(startNode);    //增加输入节点
				}
			}
		} catch (Exception e2) {
			e2.printStackTrace();
		}
		finally{
			try {
				if(br!=null){
					br.close();
					br=null;
				}
			} catch (IOException e) {
				e.printStackTrace();
			}
		}
		FileWriter writer = null;
		try {
			writer = new FileWriter(typeAndTypeIdPath);	//用File对象来构造FileWriter，写数据时，从文件开头开始写起，会覆盖以前的数据
			for(String type:type2Typeid.keySet()){  //将类型以及对应的类型id写入文件typeAndTypeIdPath中
				writer.write(type+" "+type2Typeid.get(type)+"\r\n");
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
		
		return data;
	}
}
