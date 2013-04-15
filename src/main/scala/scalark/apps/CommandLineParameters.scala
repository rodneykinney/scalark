/*
Copyright 2013 Rodney Kinney

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
package scalark.apps

case class Param(val name: String, val desc: String, val required: Boolean) {}

case class ParseException(msg: String) extends Exception(msg)

trait CommandLineParameters {
  def usage: Seq[Param]
  def optional(name: String, desc: String) = Param(name, desc, false)
  def required(name: String, desc: String) = Param(name, desc, true)

  def parse(args: Array[String]) = {
    if (args.length == 0) {
      printUsage
      false
    } else {
      try {
        val parsedParams = parseArgList(args.toList)
        var missingParams = this.usage.filter(_.required).map(_.name).toSet -- parsedParams
        if (missingParams.size > 0) {
          println("Missing required parameter:  " + missingParams.mkString(","))
          printUsage
          false
        } else
          true
      } catch {
        case e: Exception => {
          e.printStackTrace
          printUsage
          false
        }
      }
    }
  }

  def parseArgList(args: List[String]): List[String] = args match {
    case head :: tail => {
      if (!head.startsWith("-")) throw new ParseException("Unrecognized option: " + head)
      val paramName = head.substring(1)
      try {
        val field = this.getClass.getDeclaredField(paramName)
        field.setAccessible(true)
        field.getType match {
          case t if t == classOf[Boolean] => tail match {
            case "true" :: next => { field.setBoolean(this, true); paramName :: parseArgList(next) }
            case "false" :: next => { field.setBoolean(this, false); paramName :: parseArgList(next) }
            case Nil => { field.setBoolean(this, false); paramName :: Nil }
            case s :: next if s.startsWith("-") => { field.setBoolean(this, true); paramName :: parseArgList(tail) }
            case s :: next => throw new ParseException("Illegal value for boolean option: " + s)
          }
          case t if t == classOf[String] => tail match {
            case s :: next if !s.startsWith("-") => { field.set(this, tail.head); paramName :: parseArgList(next) }
            case _ => throw new ParseException("Missing value for parameter: " + paramName)
          }
          case t if t == classOf[Int] => tail match {
            case s :: next if !s.startsWith("-") => { field.setInt(this, tail.head.toInt); paramName :: parseArgList(next) }
            case _ => throw new ParseException("Missing value for parameter: " + paramName)
          }
          case t if t == classOf[Double] => tail match {
            case s :: next if !s.startsWith("-") => { field.setDouble(this, tail.head.toDouble); paramName :: parseArgList(next) }
            case _ => throw new ParseException("Missing value for parameter: " + paramName)
          }
        }
      } catch {
        case e: NoSuchFieldException => throw new ParseException("Unrecognized option: " + paramName)
      }
    }
    case Nil => Nil
  }

  def printUsage = {
    println("Options:")
    for (p <- this.usage) this.getClass.getDeclaredField(p.name).getType match {
      case t if t == classOf[Boolean] && p.required => println("  -" + p.name + ": " + p.desc)
      case t if t == classOf[Boolean] && !p.required => println("  [-" + p.name + "] : " + p.desc)
      case _ if p.required => println("  -" + p.name + " <value> : " + p.desc)
      case _ => println("  [-" + p.name + " <value>] : " + p.desc)
    }
  }
  
  override def toString = {
    val sb = new StringBuilder()
    sb.append(CommandLineParameters.this.getClass.getName().substring(CommandLineParameters.this.getClass.getName.lastIndexOf('.')+1))
    sb.append("(")
    val params = for (p <- this.usage) yield {
      val field = CommandLineParameters.this.getClass.getDeclaredField(p.name)
      field.setAccessible(true)
      p.name+"="+field.get(CommandLineParameters.this)
    }
    sb.append(params.mkString(","))
    sb.append(")")
    sb.toString
  }
}

