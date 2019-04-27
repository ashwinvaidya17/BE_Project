using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using MLAgents;
using UnityStandardAssets.Vehicles.Car;
public class CarAgent : Agent
{
    public GameObject CarPrefab;
    public CarController m_controller;
    private Vector3 forward;
    private WheelCollider fl, hl; //to get local forward direction

    public Transform []SpawnPoints;
    private int spawnIndex = 0 ;
   
    public override void InitializeAgent()
    {
        AgentReset();
    }

    public override void CollectObservations()
    {
        AddVectorObs(Vector3.Dot(forward, CarPrefab.GetComponent<Rigidbody>().velocity));
        //todo
        //get distance to goal
    }
    public override void AgentAction(float[] vectorAction, string textAction)
    {
        forward = Vector3.Normalize(fl.transform.position - hl.transform.position);
        if (CarPrefab.transform.position.y < -0.5f) // if it falls off
        {
            SetReward(-50);
            // Debug.Log(-50);
            Done();
            return;
        }
        // steer, acc, footbrake, handbrake
        m_controller.Move(vectorAction[0], vectorAction[1], vectorAction[1],0f);
       //to-do assign rewards based on velocity
       SetReward(Vector3.Dot(forward, CarPrefab.GetComponent<Rigidbody>().velocity) /5);
       if(GetReward() < 0) // penalty for going back
        SetReward(GetReward()*4);
    //    Debug.Log(Vector3.Dot(forward, CarPrefab.GetComponent<Rigidbody>().velocity) /5);
       //to-do assign reward based on proximity to goal
        
    }
    public override void AgentReset()
    {
        CarPrefab.GetComponent<Rigidbody>().velocity = Vector3.zero;
        CarPrefab.GetComponent<Rigidbody>().angularVelocity = Vector3.zero;
        fl = CarPrefab.transform.Find("WheelsHubs").Find("WheelHubFrontLeft").GetComponent<WheelCollider>();
        hl = CarPrefab.transform.Find("WheelsHubs").Find("WheelHubRearLeft").GetComponent<WheelCollider>();
        m_controller = CarPrefab.GetComponent<CarController>();
        spawnIndex = Random.Range(0,SpawnPoints.Length);
        CarPrefab.gameObject.transform.position = SpawnPoints[spawnIndex].position;
        CarPrefab.gameObject.transform.rotation = Quaternion.identity;
        forward = Vector3.Normalize(fl.transform.position - hl.transform.position);
        agentParameters.agentCameras[0] = CarPrefab.transform.Find("Camera").GetComponent<Camera>();
    }

    public override void AgentOnDone()
    {
        AgentReset();
    }
}
